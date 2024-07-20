#!/usr/bin/env python3

"""Indexing by testing of all combinations, very stupid and very slow, but extremely reliable."""

import copy
import multiprocessing.pool
import numbers
import typing
import warnings

from tqdm.autonotebook import tqdm
import psutil
import torch

from ..bragg import hkl_reciprocal_to_uq, uf_to_uq
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import compute_nb_matches, ray_phi_dist
from ..projection import detector_to_ray
from ..reciprocal import primitive_to_reciprocal
from ..rotation import omega_to_rot, rotate_crystal, uniform_so3_meshgrid
from ..symmetry import find_symmetric_rotations, reduce_omega_range


try:
    NCPU = len(psutil.Process().cpu_affinity())  # os.cpu_count() wrong on slurm
except AttributeError:  # failed on mac
    NCPU = psutil.cpu_count()


class StupidIndexator(torch.nn.Module):
    """Brute force indexation method.

    Allows you to find, slowly but extremely reliably, all the oriantations
    of the different grains of a given material.
    """

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
        self, omega: torch.Tensor, uq_exp: torch.Tensor, phi_max: float
    ) -> torch.Tensor:
        """Compute the matching rate of a sub patch."""
        uq_theo = self._compute_uq(omega)
        rate = compute_nb_matches(uq_exp, uq_theo, phi_max)
        return rate

    # @torch.compile(dynamic=False)  # failed on multithreading cuda
    def _compute_uq(self, omega: torch.Tensor) -> torch.Tensor:
        """Help for ``_compute_rate``."""
        rot = omega_to_rot(*omega.movedim(1, 0), cartesian_product=False)  # (m, 3, 3)
        reciprocal = rotate_crystal(self._reciprocal, rot, cartesian_product=False)
        uq_theo = hkl_reciprocal_to_uq(self._hkl, reciprocal)  # (h, m, 3)
        uq_theo = torch.movedim(uq_theo, 0, 1).contiguous()  # (m, h, 3)
        return uq_theo

    def clone(self) -> typing.Self:
        """Return a deep copy of self."""
        return copy.deepcopy(self)

    def forward(
        self,
        uf_or_point: torch.Tensor,
        phi_max: None | numbers.Real = None,
        omega: None | numbers.Real | torch.Tensor = None,
        *,
        poni: None | torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Return the rotations ordered by matching rate.

        Parameters
        ----------
        uf_or_point : torch.Tensor
            The 2d point associated to uf in the referencial of the detector of shape (n, 2).
            Could be also the unitary vector \(u_q\) of shape (n, 3).
        phi_max : float, optional
            The maximum positive angular distance in radian
            to consider that the rays are closed enough.
            If nothing is supplied (default), the average distance between each \(u_q\),
            and its nearset neighbour is calculated. This argument take this value.
        omega : float or arraylike, optional
            Corresponds to all the cristal oriantations to be tested.

            * If nothing is supplied (default), this argument take this value of ``phi_max``.
            * If a scalar is supplied, The \(SO(3)\) oriantation space is sampled uniformly
                so that the angle between 2 rotations is approximatively equal
                to the value of this argument, in radian. The symmetries of the crystal are
                automatically considered to reduce the size of the space to be explored.
            * The last option is to supply 3 elementary rotation angles
                \(\omega_1, \omega_2, \omega_3\), in radian, as an array of shape (n, 3).
        poni: torch.Tensor, optional
            The 6 ordered .poni calibration parameters as a tensor of shape (6,).
            Used only if ``uf_or_point`` is a point.

        Returns
        -------
        omega : torch.Tensor
            The 3 elementary rotation angles \(\omega_1, \omega_2, \omega_3\)
            sorted by decreasing matching rate, the shape is (n, 3).
        rate : torch.Tensor
            All the sorted matching rates.

        Notes
        -----
        The calculation time is independent of the number of spots in the experimental diagram.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.stupid import StupidIndexator
        >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
        >>> e_max = 20e3 * 1.60e-19  # 20 keV
        >>> indexator = StupidIndexator(lattice, e_max)
        >>>
        >>> u_f = torch.randn(100, 3)
        >>> u_f /= torch.linalg.norm(u_f, dim=1, keepdim=True)
        >>> angle, rate = indexator(u_f)
        >>>
        """
        assert isinstance(uf_or_point, torch.Tensor), uf_or_point.__class__.__name__
        assert uf_or_point.ndim == 2, uf_or_point.shape

        # convert point to uq
        if uf_or_point.shape[-1] == 2:
            assert isinstance(poni, torch.Tensor), poni.__class__.__name__
            assert poni.shape == (6,), poni.shape
            uf_or_point = detector_to_ray(uf_or_point, poni)
        elif uf_or_point.shape[-1] == 3:
            if poni is not None:
                warnings.warn(
                    "the `poni` is ignored because uf has been provided, not point",
                    RuntimeWarning,
                )
        else:
            raise ValueError("only shape 2 and 3 allow")
        u_q = uf_to_uq(uf_or_point)

        assert u_q.shape[0] >= 2, "a minimum of 2 spots are required for indexation"

        # sampling the space of the rotations
        if phi_max is None:
            phi_max = float(torch.mean(torch.sort(ray_phi_dist(u_q, u_q), dim=0).values[1, :]))
        if omega is None:
            omega = phi_max
        if isinstance(omega, numbers.Real):
            omega = uniform_so3_meshgrid(
                omega,  # check omega here
                reduce_omega_range(
                    ((-torch.pi, torch.pi), (-torch.pi/2, torch.pi/2), (-torch.pi, torch.pi)),
                    find_symmetric_rotations(self._reciprocal),
                ),
            )
        else:
            omega = torch.asarray(omega, dtype=self._reciprocal.dtype)
            assert omega.ndim == 2 and omega.shape[1] == 3, omega.shape

        # split multi GPUs
        devices = (
            [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            or [torch.device("cpu")]
        )
        models = [self.clone().to(d) for d in devices]
        # for model in models:  # compilation is not allways working with cuda and multithreading
        #     model._compute_uq = torch.compile(model._compute_uq, dynamic=False)
        omegas = [omega.to(d) for d in devices]

        # compute all rates
        batch = 512
        total = len(omega) // batch + int(bool(len(omega) % batch))
        with multiprocessing.pool.ThreadPool(max(NCPU, 2*len(devices))) as pool:
            rate = torch.cat(list(
                tqdm(
                    pool.imap(
                        lambda i: (
                            models[i % len(devices)]._compute_rate(  # pylint: disable=W0212
                                omegas[i % len(devices)][batch*i:batch*(i+1)], u_q, phi_max
                            )  # in u_q device
                        ),
                        range(total),
                    ),
                    total=total,
                    unit="simul",
                    unit_scale=batch,
                    smoothing=0.01,
                )
            ))

        # sorted result
        order = torch.argsort(rate, descending=True)
        return omega[order], rate[order]
