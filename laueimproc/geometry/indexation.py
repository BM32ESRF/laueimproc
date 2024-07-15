#!/usr/bin/env python3

"""Helper for brute force and nn indexation."""

import copy
import multiprocessing.pool
import numbers
import typing

from tqdm.autonotebook import tqdm
import psutil
import torch

from .bragg import hkl_reciprocal_to_uq, uf_to_uq, uq_to_uf
from .hkl import select_hkl
from .lattice import lattice_to_primitive
from .metric import compute_matching_rate, raydotdist
from .projection import detector_to_ray, ray_to_detector
from .reciprocal import primitive_to_reciprocal
from .rotation import angle_to_rot, rotate_crystal


NCPU = len(psutil.Process().cpu_affinity())


class NNIndexator(torch.nn.Module):
    """Indexation with neuronal network approch.

    Attributes
    ----------
    n_bins : int
        The input histogram len (readonly).
    """

    def __init__(self, n_bins: numbers.Integral, n_outputs: numbers.Integral):
        """Initialize the indexator.

        Parameters
        ----------
        n_bins : int
            The input histogram len.
        n_outputs : int
            The size of the output vector.
        """
        assert isinstance(n_bins, numbers.Integral), n_bins.__class__.__name__
        assert n_bins > 0, n_bins
        assert isinstance(n_outputs, numbers.Integral), n_outputs.__class__.__name__
        assert n_outputs > 0, n_outputs

        super().__init__()

        self._n_bins = int(n_bins)

        avg = (n_bins + n_outputs) // 2
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(n_bins, (15*avg + n_bins)//2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear((15*avg + n_bins)//2, 15*avg),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(15*avg, n_outputs),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(15*avg, n_outputs),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """Predict the hkl class from the angular histogram.

        Parameters
        ----------
        hist : torch.Tensor
            The histogram of shape (batch, h).

        Returns
        -------
        pred : torch.Tensor
            The prediction vector for each histogram, shape (batch, p).

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation import NNIndexator
        >>> model = NNIndexator(1024, 100)
        >>> hist = torch.randint(0, 10, (1024,), dtype=torch.float32)
        >>> pred = model(hist)
        >>> pred.shape
        torch.Size([100])
        >>>
        """
        return self.layer_3(self.layer_2(self.layer_1(hist)))

    @property
    def n_bins(self) -> int:
        """Return the input histogram len."""
        return self._n_bins

    def uq_to_hist(
        self, u_q: torch.Tensor, indices: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Compute the histogram of the \(u_q\) vectors.

        Parameters
        ----------
        u_q : torch.Tensor
            The \(u_q\) vectors of a single diagram of shape (n, 3).
        indices : arraylike, optional
            The index of the \(u_q\) to consider. If not provided, all are considered.

        Returns
        -------
        hist : torch.Tensor
            The float32 histograms of shape (n, self.n_bins) with n the number of histograms.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation import NNIndexator
        >>> model = NNIndexator(1024, 100)
        >>> u_q = torch.randn(2000, 3)
        >>> u_q *= torch.rsqrt(torch.sum(u_q * u_q, dim=1, keepdim=True))
        >>> model.uq_to_hist(u_q).shape
        torch.Size([2000, 1024])
        >>> model.uq_to_hist(u_q, indices=torch.tensor([0, 1, 4, 6])).shape
        torch.Size([4, 1024])
        >>>
        """
        assert isinstance(u_q, torch.Tensor), u_q.__class__.__name__
        assert u_q.ndim == 2 and u_q.shape[1] == 3, u_q.shape

        if indices is None:
            uq_ref = u_q
        else:
            indices = torch.asarray(indices, dtype=torch.int64, device=u_q.device)
            uq_ref = u_q[indices]

        dist = torch.acos(raydotdist(uq_ref, u_q))  # angle distance matrix
        res = 0.5 * (torch.pi / 180.0)  # angle reolution
        hist = torch.func.vmap(torch.histc)(
            dist, bins=self._n_bins, min=0.0, max=(self._n_bins+1)*res
        )
        return hist


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
            select_hkl(reciprocal=self._reciprocal, e_max=e_max),
            requires_grad=False,
        )

    def _compute_rate(
        self, angles: torch.Tensor, uq_exp: torch.Tensor, angle_max_matching: float
    ) -> torch.Tensor:
        """Compute the matching rate of a sub patch."""
        uq_theo = self._compute_uq(angles)
        rate = compute_matching_rate(uq_exp, uq_theo, angle_max_matching)
        return rate

    @torch.compile(dynamic=False)
    def _compute_uq(self, angles: torch.Tensor) -> torch.Tensor:
        """Help for ``_compute_rate``."""
        rot = angle_to_rot(*angles.movedim(1, 0), cartesian_product=False)  # (m, 3, 3)
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
        angle_res: None | numbers.Real = None,
        angle_max_matching: None | numbers.Real = None,
    ) -> torch.Tensor:
        r"""Return the rotations ordered by matching rate.

        Parameters
        ----------
        u_q : torch.Tensor
            The experimental \(u_q\) vectors in the lab base \(\mathcal{B^l}\).
            For one diagram only, shape (q, 3).
        angle_res : float, optional
            Angular resolution of explored space in radian.
            By default, it is choosen by a statistic heuristic.
        angle_max_matching : float, optional
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
        >>> from laueimproc.geometry.indexation import StupidIndexator
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
            angle_res = 0.5 * float(
                torch.mean(
                    torch.acos(
                        torch.sort(raydotdist(u_q, u_q), descending=True, dim=0).values[1, :]
                    )
                )
            )
        else:
            assert isinstance(angle_res, numbers.Real), angle_res.__class__.__name__
            assert angle_res > 0, angle_res
        if angle_max_matching is None:
            angle_max_matching = angle_res
        else:
            assert isinstance(angle_max_matching, numbers.Real), \
                angle_max_matching.__class__.__name__
            assert angle_max_matching > 0, angle_max_matching

        # initialisation
        angle = torch.meshgrid(
            torch.arange(-torch.pi/4, torch.pi/4, angle_res, device=u_q.device, dtype=u_q.dtype),
            torch.arange(-torch.pi/4, torch.pi/4, angle_res, device=u_q.device, dtype=u_q.dtype),
            torch.arange(-torch.pi/4, torch.pi/4, angle_res, device=u_q.device, dtype=u_q.dtype),
            indexing="ij",
        )
        angle = torch.cat([a.ravel().unsqueeze(-1) for a in angle], dim=1)  # (n, 3)

        # split multi GPUs
        devices = (
            [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            or [torch.device("cpu")]
        )
        models = [self.clone().to(d) for d in devices]
        angles = [angle.to(d) for d in devices]

        # compute all rates
        batch = 128
        total = len(angle) // batch + int(bool(len(angle) % batch))
        with multiprocessing.pool.ThreadPool(max(NCPU, 2*len(devices))) as pool:
            rate = torch.cat(list(
                tqdm(
                    pool.imap(
                        lambda i: (
                            models[i % len(devices)]._compute_rate(  # pylint: disable=W0212
                                angles[i % len(devices)][batch*i:batch*(i+1)],
                                u_q,
                                angle_max_matching,
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


class UqGenerator:
    """Generate random labelised histograms."""

    def __init__(
        self,
        lattice: torch.Tensor,
        e_max: numbers.Real,
        poni: torch.Tensor,
        cam_size: tuple[float, float],
    ):
        """Initialize histogram generator.

        Parameters
        ----------
        lattice : torch.Tensor
            The relaxed lattice parameters of multiple grains, of shape (n, 6).
        e_max : float
            The maximum energy of the beam line in J.
        poni : torch.Tensor
            The 6 ordered .poni calibration parameters as a tensor of shape (6,)
        cam_size : tuple[float, float]
            The size of the detector in m.
        """
        assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
        assert lattice.ndim == 2 and lattice.shape[1] == 6, lattice.shape
        assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
        assert 0.0 < e_max < torch.inf, e_max
        assert isinstance(poni, torch.Tensor), poni.__class__.__name__
        assert poni.shape == (6,), poni.shape
        assert hasattr(cam_size, "__iter__"), cam_size.__class__.__name__
        cam_size = list(cam_size)
        assert len(cam_size) == 2, len(cam_size)
        assert isinstance(cam_size[0], numbers.Real) and isinstance(cam_size[1], numbers.Real)
        assert cam_size[0] > 0 and cam_size[1] > 0, cam_size

        self._reciprocal = primitive_to_reciprocal(lattice_to_primitive(lattice))
        self._hkl = select_hkl(reciprocal=self._reciprocal, e_max=e_max)
        self._poni = poni
        self._cam_size = (float(cam_size[0]), float(cam_size[1]))

    def draw_uq(self):
        r"""Simulate a random \(u_q\).

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation import UqGenerator
        >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
        >>> e_max = 25e3 * 1.60e-19  # 25 keV
        >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
        >>> cam_size = (0.1468, 0.1468)
        >>> generator = UqGenerator(lattice[None, :], e_max, poni, cam_size)
        >>> u_q, hkl, classes = generator.draw_uq()
        >>>
        """
        # simulation
        nb_rot = 3
        rot1 = 2.0 * torch.pi * torch.rand(nb_rot) - torch.pi
        rot2 = torch.pi * torch.rand(nb_rot) - 0.5 * torch.pi
        rot3 = 2.0 * torch.pi * torch.rand(nb_rot) - torch.pi
        rot = angle_to_rot(rot1, rot2, rot3, cartesian_product=False)  # (r, 3, 3)
        reciprocal = rotate_crystal(self._reciprocal, rot)  # (m, r, 3, 3)
        u_q = hkl_reciprocal_to_uq(self._hkl, reciprocal)  # (n, m, r, 3)
        u_f = uq_to_uf(u_q)
        point, dist = ray_to_detector(u_f, self._poni)  # (n, m, r, 2)

        # selection
        cond = (
            (dist > 0)
            & (point[..., 0] >= 0) & (point[..., 1] >= 0)
            & (point[..., 0] <= self._cam_size[0]) & (point[..., 1] <= self._cam_size[1])
        )
        hkl = self._hkl.reshape(-1, 1, 1, 3).expand(-1, len(self._reciprocal), nb_rot, -1)
        classes = (
            torch.arange(len(self._reciprocal))
            .reshape(1, -1, 1)
            .expand(len(self._hkl), -1, nb_rot)
        )
        point, hkl, classes = point[cond], hkl[cond], classes[cond]

        # dataaug
        point += min(self._cam_size) * 1e-3 * torch.randn_like(point)  # add position noise
        cond = torch.rand(point.shape[:-1]) < 0.05  # add false detection
        point[cond] = torch.rand_like(point[cond]) * torch.tensor(self._cam_size)
        hkl[cond] = 0
        classes[cond] = -1

        # reverse operation
        u_f = detector_to_ray(point, self._poni)
        u_q = uf_to_uq(u_f)

        return u_q, hkl, classes

    @staticmethod
    def hkl_to_family(hkl: torch.Tensor) -> torch.Tensor:
        """Find the family of each hkl."""
        assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
        assert hkl.shape[-1] == 3, hkl.shape

        # all_families = torch.tensor(list(itertools.product(range(5), range(5), range(5))))
        # all_families = all_families[
        #     torch.gcd(torch.gcd(all_families[:, 0], all_families[:, 1]), all_families[:, 2]) == 1
        # ]
        # all_families = torch.tensor([
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [1, 0, 0],
        #     [0, 1, 1],
        #     [1, 1, 0],
        #     [1, 0, 1],
        #     [1, 1, 1],
        # ])

        # family = hkl // torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2])
