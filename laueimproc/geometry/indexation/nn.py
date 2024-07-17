#!/usr/bin/env python3

"""Prediction of hkl families with a neural network."""

import numbers
import typing

import torch

from ..bragg import hkl_reciprocal_to_uq_energy, uq_to_uf
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import ray_cosine_dist
from ..projection import ray_to_detector
from ..reciprocal import primitive_to_reciprocal
from ..rotation import rand_rot, rotate_crystal
from ..symmetry import get_hkl_family_member


class NNIndexator(torch.nn.Module):
    """Indexation with neuronal network approch.

    Attributes
    ----------
    bins : int
        The input histogram len (readonly).
    families : list[list[tuple[int, int, int]]]
        The hkl families for each material.
    """

    def __init__(self, lattice: torch.Tensor, **kwargs):
        """Initialize the indexator.

        Parameters
        ----------
        lattice : torch.Tensor
            The relaxed lattice parameters of all the materials, of shape (n, 6).
        hkl_max : list[int], default=[10, ...]
            The maximum absolute hkl sum for each material such as |h| + |k| + |l| <= hkl_max.
        phi_res : float, default=0.1*pi/180
            Angular resolution of the input histogram (in radian).
        phi_max : float, default=2*pi/3
            The maximum angle of the histogram (in radian).
        """
        assert set(kwargs).issubset(
            {"lattice", "hkl_max", "n_grain", "phi_res", "phi_max"}
        ), kwargs

        assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
        assert lattice.ndim == 2 and lattice.shape[0] >= 1 and lattice.shape[1] == 6, lattice.shape

        self._attrs = {}
        self._attrs["hkl_max"] = kwargs.get("hkl_max", 8)
        assert isinstance(self._attrs["hkl_max"], numbers.Integral), self._attrs["hkl_max"]
        assert 0 < self._attrs["hkl_max"] <= torch.iinfo(torch.int16).max, self._attrs["hkl_max"]
        self._attrs["phi_res"] = kwargs.get("phi_res", 0.1 * torch.pi/180)
        assert isinstance(self._attrs["phi_res"], numbers.Real), self._attrs["phi_res"]
        assert 0 < self._attrs["phi_res"] <= torch.pi/2, self._attrs["phi_res"]
        self._attrs["phi_max"] = kwargs.get("phi_max", 120.0 * torch.pi/180)
        assert isinstance(self._attrs["phi_max"], numbers.Real), self._attrs["phi_max"]
        assert self._attrs["phi_max"] <= self._attrs["phi_max"], self._attrs["phi_max"]

        super().__init__()

        # precompute contants
        self._reciprocal = torch.nn.Parameter(
            primitive_to_reciprocal(lattice_to_primitive(lattice)),
            requires_grad=False,
        )
        self._attrs["families"] = [
            self.find_families_indices(
                self._attrs["hkl_max"], reciprocal
            )
            for reciprocal in self._reciprocal
        ]

        # main neuronal model
        avg = (self.bins + sum(map(len, self._attrs["families"]))) // 2
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(self.bins, (15*avg + self.bins)//2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear((15*avg + self.bins)//2, 15*avg),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(15*avg, 15*avg),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        self.layer_hlk = torch.nn.Sequential(
            torch.nn.Linear(15*avg, 1 + sum(map(len, self._attrs["families"]))),
            torch.nn.Softmax(dim=-1)
        )

    @staticmethod
    def add_noise(
        point: torch.Tensor,
        hkl: torch.Tensor,
        material: torch.Tensor,
        miss_detection_rate: numbers.Real = 0.2,
        noise_std: numbers.Real | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation to a very pretty diagram.

        Parameters
        ----------
        point : torch.Tensor
            The points on the camera of all the grains, shape (n, 2).
        hkl : torch.Tensor
            The full hkl list, shape (n, 3).
        material : torch.Tensor
            The list of the index of the material, shape (n,).
        miss_detection_rate : float, default=20%
            The relative number of spots to be moved anywhere.
        noise_std : float, optional
            The standard deviation of the position uncertaincy, in meter.

        Returns
        -------
        point, hkl, material : torch.Tensor
            A reference to the input parameters.

        Notes
        -----
        Change the data inplace.
        """
        assert isinstance(point, torch.Tensor), point.__class__.__name__
        assert point.ndim == 2 and point.shape[1] == 2, point.shape
        assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
        assert hkl.ndim == 2 and hkl.shape[1] == 3, hkl.shape
        assert isinstance(material, torch.Tensor), material.__class__.__name__
        assert material.ndim == 1, material.shape
        assert point.shape[0] == hkl.shape[0] == material.shape[0]

        amin, amax = torch.aminmax(point, dim=0).tolist()
        size = amax - amin

        if noise_std is None:
            noise_std = 1e-3 * float(size.mean())
        else:
            assert isinstance(noise_std, numbers.Real), noise_std.__class__.__name__
            assert noise_std >= 0.0, noise_std
        assert isinstance(miss_detection_rate, numbers.Real), miss_detection_rate.__class__.__name__
        assert 0.0 <= miss_detection_rate < 1.0, miss_detection_rate

        point += noise_std * torch.randn_like(point)  # add position noise
        miss = torch.rand_like(point[:, 0]) < miss_detection_rate  # add false detection
        point[miss] = torch.rand_like(point[miss]) * (size + amin)[:, None]
        hkl[miss], material[miss] = 0, -1

        return point, hkl, material

    @property
    def bins(self) -> int:
        """Return the input histogram len."""
        return round(self._attrs["phi_max"] / self._attrs["phi_res"])

    @property
    def families(self) -> list[list[tuple[int, int, int]]]:
        """Return the hkl families for each material."""
        return self._attrs["families"].copy()  # protection aginst user

    @staticmethod
    def find_families_indices(
        hkl_max: numbers.Integral, reciprocal: torch.Tensor
    ) -> list[tuple[int, int, int]]:
        r"""Find all hkl families.

        Parameters
        ----------
        hkl_max : int
            The maximum absolute hkl sum such as |h| + |k| + |l| <= hkl_max.
        reciprocal : torch.Tensor
            Matrix \(\mathbf{B}\) in any orthonormal base, of shape (3, 3).
            It is used to find the symmetries.

        Returns
        -------
        all_families : torch.Tensor
            The list of canonical members for all families.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
        >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal
        >>> primitive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> NNIndexator.find_families_indices(4, primitive_to_reciprocal(primitive))
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (3, 1, 0)]
        >>> NNIndexator.find_families_indices(3, primitive_to_reciprocal(primitive))
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)]
        >>> primitive = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> NNIndexator.find_families_indices(3, primitive_to_reciprocal(primitive))
        [(0, 1, 0), (0, 1, 1), (0, 2, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 0), (2, 1, 0)]
        >>>
        """
        assert isinstance(hkl_max, numbers.Integral), hkl_max.__class__.__name__
        assert 0 < hkl_max <= torch.iinfo(torch.int16).max, hkl_max
        assert isinstance(reciprocal, torch.Tensor), reciprocal.__class__.__name__
        assert reciprocal.shape == (3, 3), reciprocal.shape

        hkl = select_hkl(hkl_max=hkl_max)
        hkl = get_hkl_family_member(hkl, reciprocal)
        hkl = sorted(set(map(tuple, hkl.tolist())))
        return hkl

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
        """
        return self.layer_hlk(self.layer_3(self.layer_2(self.layer_1(hist))))

    def simulate_random_perfect_diagram(
        self,
        poni: torch.Tensor,
        cam_size: tuple[float, float],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate a perfect multigrain laue diagram.

        Parameters
        ----------
        poni : torch.Tensor
            The 6 ordered .poni calibration parameters as a tensor of shape (6,).
        cam_size : tuple[float, float]
            The size of the detector (in meter).
        e_min : float, default=5keV
            The minimum energy of the beam line in J.
        e_max : float, default=25keV
            The maximum energy of the beam line in J.
        nbr_grain : int, default=3
            The number of simulated grains.

        Returns
        -------
        point : torch.Tensor
            The points on the camera of all the grains, shape (n, 2).
        hkl : torch.Tensor
            The full hkl list, shape (n, 3).
        material : torch.Tensor
            The list of the index of the material, shape (n,).
        energy : torch.Tensor
            The energy in J of each spot, it contains the harmonics, shape (n,).

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
        >>> lattice = torch.tensor([
        ...     [3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2],
        ... ])
        >>> indexator = NNIndexator(lattice)
        >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
        >>> cam_size = (0.146, 0.146)
        >>> e_min, e_max = 5e3 * 1.60e-19, 25e3 * 1.60e-19  # 5-25 keV
        >>> point, hkl, material, energy = indexator.simulate_random_perfect_diagram(
        ...     poni, cam_size, e_min=e_min, e_max=e_max
        ... )
        >>>
        """
        assert isinstance(poni, torch.Tensor), poni.__class__.__name__
        assert poni.shape == (6,), poni.shape
        assert hasattr(cam_size, "__iter__"), cam_size.__class__.__name__
        cam_size = list(cam_size)
        assert len(cam_size) == 2, len(cam_size)
        assert isinstance(cam_size[0], numbers.Real) and isinstance(cam_size[1], numbers.Real)
        assert cam_size[0] > 0 and cam_size[1] > 0, cam_size
        assert set(kwargs).issubset({"e_min", "e_max", "nbr_grain"}), kwargs
        e_min_e_max = (kwargs.get("e_min", 5e3*1.6e-19), kwargs.get("e_max", 25e3*1.6e-19))
        assert isinstance(e_min_e_max[0], numbers.Real), e_min_e_max[0].__class__.__name__
        assert 0.0 <= e_min_e_max[0], e_min_e_max[0]
        assert isinstance(e_min_e_max[1], numbers.Real), e_min_e_max[1].__class__.__name__
        assert e_min_e_max[0] <= e_min_e_max[1] < torch.inf, e_min_e_max[1]
        nbr_grain = kwargs.get("nbr_grain", 3)
        assert isinstance(nbr_grain, numbers.Integral), nbr_grain.__class__.__name__
        assert nbr_grain > 0, nbr_grain

        # initial conditions
        material = torch.randint(
            0, len(self._reciprocal), (nbr_grain,), device=self._reciprocal.device
        )
        rot = rand_rot(nbr_grain, dtype=self._reciprocal.dtype, device=self._reciprocal.device)

        # simulation
        reciprocal = self._reciprocal[material]  # (grain, 3, 3)
        reciprocal = rotate_crystal(self._reciprocal, rot, cartesian_product=False)  # (grain, 3, 3)
        hkl = select_hkl(reciprocal=self._reciprocal, e_max=e_min_e_max[1])  # multi material ok
        ray, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal)  # (n, grain, 3)
        cond = energy >= e_min_e_max[0]  # (n, grain)
        ray, energy = ray[cond], energy[cond]  # (n', 3)
        material = material[None, :].expand(cond.shape)[cond]  # (n',)
        hkl = hkl[:, None, :].expand(*cond.shape, -1)[cond]  # (n', 3)
        ray = uq_to_uf(ray)  # (n', 3)
        point, dist = ray_to_detector(ray, poni)  # (n', 2), because no poni batch

        # rejection
        cond = dist > 0
        point, hkl, material, energy = point[cond], hkl[cond], material[cond], energy[cond]
        cond = (  # select point in camera
            (point[..., 0] >= 0) & (point[..., 1] >= 0)
            & (point[..., 0] <= cam_size[0]) & (point[..., 1] <= cam_size[1])
        )
        point, hkl, material, energy = point[cond], hkl[cond], material[cond], energy[cond]

        return point, hkl, material, energy

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
        """
        assert isinstance(u_q, torch.Tensor), u_q.__class__.__name__
        assert u_q.ndim == 2 and u_q.shape[1] == 3, u_q.shape

        if indices is None:
            uq_ref = u_q
        else:
            indices = torch.asarray(indices, dtype=torch.int64, device=u_q.device)
            uq_ref = u_q[indices]

        dist = torch.acos(ray_cosine_dist(uq_ref, u_q))  # angle distance matrix
        hist = torch.func.vmap(torch.histc)(
            dist, bins=self.bins, min=0.0, max=self._attrs["phi_max"]
        )
        return hist
