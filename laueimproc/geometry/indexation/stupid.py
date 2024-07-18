#!/usr/bin/env python3

"""Indexing by testing of all combinations, very stupid and very slow, but extremely reliable."""

import copy
import multiprocessing.pool
import numbers
import typing

from tqdm.autonotebook import tqdm
import psutil
import torch

from ..bragg import hkl_reciprocal_to_uq
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import compute_matching_rate, ray_phi_dist
from ..reciprocal import primitive_to_reciprocal
from ..rotation import omega_to_rot, rotate_crystal


try:
    NCPU = len(psutil.Process().cpu_affinity())  # os.cpu_count() wrong on slurm
except AttributeError:
    NCPU = psutil.cpu_count()


class StupidIndexator(torch.nn.Module):
    """Brute force indexation method."""

    def __init__(self, lattice: torch.Tensor, e_max: numbers.Real):
        """Initialize the indexator.

        Parameters
        ----------
        lattice : torch.Tensor
            The relaxed lattice parameters of a single grain, of shape (6,).
        e_max : float
            The maximum energy of the beam line in J.
        """
        assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
        assert lattice.shape == (6,), lattice.shape
        assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
        assert 0.0 < e_max < torch.inf, e_max

        super().__init__()

        self._reciprocal = torch.nn.Parameter(
            primitive_to_reciprocal(lattice_to_primitive(lattice)),
            requires_grad=False,
        )
        self._hkl = torch.nn.Parameter(
            select_hkl(self._reciprocal, e_max=e_max),
            requires_grad=False,
        )

    def _compute_rate(
        self, angles: torch.Tensor, uq_exp: torch.Tensor, phi_max: float
    ) -> torch.Tensor:
        """Compute the matching rate of a sub patch."""
        uq_theo = self._compute_uq(angles)
        rate = compute_matching_rate(uq_exp, uq_theo, phi_max)
        return rate

    # @torch.compile(dynamic=False)
    def _compute_uq(self, angles: torch.Tensor) -> torch.Tensor:
        """Help for ``_compute_rate``."""
        rot = omega_to_rot(*angles.movedim(1, 0), cartesian_product=False)  # (m, 3, 3)
        reciprocal = rotate_crystal(self._reciprocal, rot, cartesian_product=False)
        uq_theo = hkl_reciprocal_to_uq(self._hkl, reciprocal)  # (h, m, 3)
        uq_theo = torch.movedim(uq_theo, 0, 1).contiguous()  # (m, h, 3)
        return uq_theo

    def clone(self) -> typing.Self:
        """Return a deep copy of self."""
        return copy.deepcopy(self)

    def forward(
        self,
        u_q: torch.Tensor,
        angle: None | torch.Tensor = None,
        angle_res: None | numbers.Real = None,
        phi_max: None | numbers.Real = None,
    ) -> torch.Tensor:
        r"""Return the rotations ordered by matching rate.

        Parameters
        ----------
        u_q : torch.Tensor
            The experimental \(u_q\) vectors in the lab base \(\mathcal{B^l}\).
            For one diagram only, shape (q, 3).
        angle : torch.Tensor, optional
            If provided, these angles are going to be used. Shape (n, 3).
        angle_res : float, optional
            Angular resolution of explored space in radian.
            By default, it is choosen by a statistic heuristic.
        phi_max : float, optional
            The maximum positive angular distance in radian
            to consider that the rays are closed enough.
            If not provide, the value is the same as ``angle_res``.

        Returns
        -------
        angle : torch.Tensor
            The 3 elementary rotation angles sorted by decreasing matching rate.
            The shape is (n, 3).
        rate : torch.Tensor
            All the sorted matching rates.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.stupid import StupidIndexator
        >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
        >>> e_max = 20e3 * 1.60e-19  # 20 keV
        >>> indexator = StupidIndexator(lattice, e_max)
        >>>
        >>> u_q = torch.randn(100, 3)
        >>> u_q *= torch.rsqrt(torch.sum(u_q * u_q, dim=1, keepdim=True))
        >>> angle, rate = indexator(u_q)
        >>>
        """
        assert isinstance(u_q, torch.Tensor), u_q.__class__.__name__
        assert u_q.ndim == 2 and u_q.shape[1] == 3, u_q.shape
        assert u_q.shape[0] >= 2, "a minimum of 2 spots are required for indexation"

        # get default values
        if angle_res is None:
            angle_res = float(torch.mean(torch.sort(ray_phi_dist(u_q, u_q), dim=0).values[1, :]))
        else:
            assert isinstance(angle_res, numbers.Real), angle_res.__class__.__name__
            assert angle_res > 0, angle_res
        if phi_max is None:
            phi_max = angle_res
        else:
            assert isinstance(phi_max, numbers.Real), \
                phi_max.__class__.__name__
            assert phi_max > 0, phi_max

        # initialisation
        if angle is None:
            angle = torch.pi / 2
            angle = torch.meshgrid(  # we can do better, of course
                torch.arange(-angle, angle, angle_res, device=u_q.device, dtype=u_q.dtype),
                torch.arange(-angle/2, angle/2, angle_res, device=u_q.device, dtype=u_q.dtype),
                torch.arange(-angle, angle, angle_res, device=u_q.device, dtype=u_q.dtype),
                indexing="ij",
            )
            angle = torch.cat([a.ravel().unsqueeze(-1) for a in angle], dim=1)  # (n, 3)
        else:
            assert isinstance(angle, torch.Tensor), angle.__class__.__name__
            assert angle.ndim == 2 and angle.shape[1] == 3, angle.shape

        # split multi GPUs
        devices = (
            [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            or [torch.device("cpu")]
        )
        models = [self.clone().to(d) for d in devices]
        angles = [angle.to(d) for d in devices]

        # compute all rates
        batch = 512
        total = len(angle) // batch + int(bool(len(angle) % batch))
        with multiprocessing.pool.ThreadPool(max(NCPU, 2*len(devices))) as pool:
            rate = torch.cat(list(
                tqdm(
                    pool.imap(
                        lambda i: (
                            models[i % len(devices)]._compute_rate(  # pylint: disable=W0212
                                angles[i % len(devices)][batch*i:batch*(i+1)],
                                u_q,
                                phi_max,
                            )  # in u_q device
                        ),
                        range(total),
                    ),
                    total=total,
                    unit_scale=batch,
                )
            ))

        # sorted result
        order = torch.argsort(rate, descending=True)
        return angle[order], rate[order]
