#!/usr/bin/env python3

"""Prediction of hkl families with a neural network."""

import numbers
import typing

import torch

from ..bragg import hkl_reciprocal_to_uq, uf_to_uq, uq_to_uf
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import ray_cosine_dist
from ..projection import detector_to_ray, ray_to_detector
from ..reciprocal import primitive_to_reciprocal
from ..rotation import rand_rot, rotate_crystal


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
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
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
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
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

        dist = torch.acos(ray_cosine_dist(uq_ref, u_q))  # angle distance matrix
        res = 0.5 * torch.pi/180  # angle reolution
        hist = torch.func.vmap(torch.histc)(
            dist, bins=self._n_bins, min=0.0, max=(self._n_bins+1)*res
        )
        return hist


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
        >>> from laueimproc.geometry.indexation.nn import UqGenerator
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
        rot = rand_rot(nb_rot, dtype=self._reciprocal.dtype, device=self._reciprocal.device)

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
