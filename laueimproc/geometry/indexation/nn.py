#!/usr/bin/env python3

"""Prediction of hkl families with a neural network."""

import numbers
import warnings

import torch

from ..bragg import hkl_reciprocal_to_uq_energy, uf_to_uq, uq_to_uf
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import ray_cosine_dist
from ..projection import detector_to_ray, ray_to_detector
from ..reciprocal import primitive_to_reciprocal
from ..rotation import rand_rot, rotate_crystal
from ..symmetry import get_hkl_family_members


class NNIndexator(torch.nn.Module):
    """Indexation with neuronal network approch.

    Attributes
    ----------
    bins : int
        The input histogram len (readonly).
    families : list[list[tuple[int, int, int]]]
        The hkl families for each material.
    weights : list[torch.Tensor]
        The trainaible neuronal network model weights.
    """

    def __init__(self, lattice: torch.Tensor, **kwargs):
        """Initialize the indexator.

        Parameters
        ----------
        lattice : torch.Tensor
            The relaxed lattice parameters of all the materials, of shape (n, 6).
        hkl_max : int, default=10
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

        # precompute contants and hash tables
        self._reciprocal = torch.nn.Parameter(
            primitive_to_reciprocal(lattice_to_primitive(lattice)),
            requires_grad=False,
        )
        hkl = select_hkl(hkl_max=self._attrs["hkl_max"])
        material_hkl_2_family = {  # .[material][hkl] = family
            mat: {
                tuple(hkl_): max(map(tuple, members))
                for hkl_, members in zip(
                    hkl.tolist(), get_hkl_family_members(hkl, reciprocal).tolist()
                )
            }
            for mat, reciprocal in enumerate(self._reciprocal)
        }
        family_idx = 1
        self._attrs["material_family_to_idx"] = {}  # .[(material, family)] = family_idx
        self._attrs["families"] = []  # .[material] = families
        for mat in range(len(self._reciprocal)):
            self._attrs["families"].append(sorted(set(material_hkl_2_family[mat].values())))
            for family in self._attrs["families"][-1]:
                self._attrs["material_family_to_idx"][(mat, family)] = family_idx
                family_idx += 1
        self._families = torch.nn.Parameter(
            torch.asarray(
                sum(self._attrs["families"], start=[(0, 0, 0)]),
                device=self._reciprocal.device,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        self._attrs["hkl_hist"] = torch.zeros(
            len(self._families), dtype=torch.int64, device=self._reciprocal.device
        )
        self._attrs["cache_hist"] = torch.empty(
            0, self.bins, dtype=self._reciprocal.dtype, device=self._reciprocal.device
        )
        self._attrs["cache_hkl"] = torch.empty(
            0, len(self._families), dtype=torch.int64, device=self._reciprocal.device
        )

        # main neuronal model (slowest part of initialization)
        avg = (self.bins + len(self._families)) // 2
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
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Linear(15*avg, len(self._families)),
            torch.nn.Softmax(dim=-1)
        )

    @staticmethod
    def _add_noise(
        point: torch.Tensor,
        hkl: torch.Tensor,
        material: torch.Tensor,
        miss_detection_rate: numbers.Real = 0.2,
        noise_std: None | numbers.Real = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation to a very pretty diagram."""
        assert isinstance(point, torch.Tensor), point.__class__.__name__
        assert point.ndim == 2 and point.shape[1] == 2, point.shape
        assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
        assert hkl.ndim == 2 and hkl.shape[1] == 3, hkl.shape
        assert isinstance(material, torch.Tensor), material.__class__.__name__
        assert material.ndim == 1, material.shape
        assert point.shape[0] == hkl.shape[0] == material.shape[0]

        amin, amax = torch.aminmax(point, dim=0)
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
        point[miss] = torch.rand_like(point[miss]) * (size + amin)
        hkl[miss], material[miss] = 0, -1

        return point, hkl, material

    def _generate_perfect_diagram(
        self,
        poni: torch.Tensor,
        cam_size: tuple[float, float],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate a perfect multigrain laue diagram."""
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

    def _point_hkl_to_trainable(
        self, point: torch.Tensor, hkl: torch.Tensor, material: torch.Tensor, poni: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the data into in and out for training."""
        # convert hkl into one hot family
        low_hkl = torch.sum(torch.abs(hkl), dim=1) <= self._attrs["hkl_max"]  # (n,)
        material_family_to_idx = self._attrs["material_family_to_idx"]
        family_idx = torch.asarray([  # (n',)
            material_family_to_idx.get((i, tuple(hkl)), 0)
            for i, hkl in zip(material[low_hkl].tolist(), hkl[low_hkl].tolist())
        ], dtype=torch.int64, device=hkl.device)
        family_idx = torch.nn.functional.one_hot(family_idx, len(self._families))  # (n', f)

        # get historgams
        u_q = uf_to_uq(detector_to_ray(point, poni))
        hist = self._uq_to_hist(u_q, indices=low_hkl)  # (n', n_bins)

        return hist, family_idx

    def _uq_to_hist(
        self, u_q: torch.Tensor, *, indices: None | torch.Tensor = None
    ) -> torch.Tensor:
        r"""Compute the histogram of the \(u_q\) vectors."""
        if indices is None:
            uq_ref = u_q
        else:
            indices = torch.asarray(indices, device=u_q.device)
            uq_ref = u_q[indices]
        dist = torch.acos(ray_cosine_dist(uq_ref, u_q))  # angle distance matrix
        hist = torch.func.vmap(torch.histc)(
            dist, bins=self.bins, min=0.0, max=self._attrs["phi_max"]
        )
        return hist

    def generate_diagram(
        self, poni: torch.Tensor, cam_size: tuple[float, float], **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate a perfect multi grain noisy diagram.

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
        miss_detection_rate : float, default=20%
            The relative number of spots to be moved anywhere.
        noise_std : float, optional
            The standard deviation of the position uncertaincy, in meter.

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
        >>> point, hkl, material, energy = indexator.generate_diagram(
        ...     poni=poni, cam_size=cam_size, e_min=e_min, e_max=e_max
        ... )
        >>>
        """
        kwargs_1 = {"e_min", "e_max", "nbr_grain"}
        kwargs_2 = {"miss_detection_rate", "noise_std"}
        assert set(kwargs).issubset(kwargs_1 | kwargs_2), kwargs
        point, hkl, material, energy = self._generate_perfect_diagram(
            poni, cam_size, **{k: v for k, v in kwargs.items() if k in kwargs_1}
        )
        point, hkl, material = self._add_noise(
            point, hkl, material, **{k: v for k, v in kwargs.items() if k in kwargs_2}
        )
        return point, hkl, material, energy

    def generate_training_batch(
        self,
        poni: torch.Tensor,
        cam_size: tuple[float, float],
        batch: numbers.Integral = 128,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random batch of training data.

        Parameters
        ----------
        poni, cam_size, **kwargs
            Transmitted to ``laueimproc.geometry.indexation.nn.NNIndexator.generate_diagram``.
        batch : int, default=128
            The batch dimension, numbers of histograms returned.

        Returns
        -------
        hist : torch.Tensor
            The float histograms of shape (batch, self.bins) with n the number of histograms.
        hkl_family_onehot : torch.Tensor
            The hkl family in one hot float encoding, shape (batch, f).

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
        >>> lattice = torch.tensor([
        ...     [3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2],
        ... ])
        >>> indexator = NNIndexator(lattice, phi_max=torch.pi/6)
        >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
        >>> cam_size = (0.146, 0.146)
        >>> e_min, e_max = 5e3 * 1.60e-19, 25e3 * 1.60e-19  # 5-25 keV
        >>> hist, target = indexator.generate_training_batch(
        ...     poni=poni, cam_size=cam_size, e_min=e_min, e_max=e_max, batch=8
        ... )
        >>>
        >>> # try to overfit
        >>> loss = torch.nn.CrossEntropyLoss()
        >>> loss_init = float(loss(indexator(hist), target))
        >>> optim = torch.optim.RAdam(indexator.weights, weight_decay=0.0005)
        >>> for _ in range(5):
        ...     optim.zero_grad()
        ...     output = loss(indexator(hist), target)
        ...     output.backward()
        ...     optim.step()
        ...
        >>> loss_final = float(loss(indexator(hist), target))
        >>> loss_init > loss_final
        True
        >>>
        """
        assert isinstance(batch, numbers.Integral), batch.__class__.__name__
        assert batch > 0, batch

        # generate data
        while len(self._attrs["cache_hist"]) < batch:
            point, hkl, material, _ = self.generate_diagram(poni, cam_size, **kwargs)
            hist, hkl = self._point_hkl_to_trainable(point, hkl, material, poni)
            self._attrs["cache_hist"] = torch.cat([self._attrs["cache_hist"].to(hist.device), hist])
            self._attrs["cache_hkl"] = torch.cat([self._attrs["cache_hkl"].to(hkl.device), hkl])

        # trunc data
        hist = self._attrs["cache_hist"][:batch]
        self._attrs["cache_hist"] = self._attrs["cache_hist"][batch:]
        hkl = self._attrs["cache_hkl"][:batch]
        self._attrs["cache_hkl"] = self._attrs["cache_hkl"][batch:]

        # update data repartition
        self._attrs["hkl_hist"] = self._attrs["hkl_hist"].to(hkl.device)
        self._attrs["hkl_hist"] += torch.sum(hkl, dim=0)

        return hist, hkl.to(hist.dtype)

    @property
    def bins(self) -> int:
        """Return the input histogram len."""
        return round(self._attrs["phi_max"] / self._attrs["phi_res"])

    @property
    def families(self) -> list[list[tuple[int, int, int]]]:
        """Return the hkl families for each material."""
        return self._attrs["families"].copy()  # protection aginst user

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
        return self.layer_4(self.layer_3(self.layer_2(self.layer_1(hist))))

    def predict_hkl(
        self,
        uf_or_point: torch.Tensor,
        *,
        poni: None | torch.Tensor = None,
        indices: None | torch.Tensor = None,
    ) -> tuple[int, tuple[int, int, int]]:
        r"""Predict hkl index class and material.

        Parameters
        ----------
        uf_or_point : torch.Tensor
            The 2d point associated to uf in the referencial of the detector of shape (n, 2).
            Could be also the unitary vector \(u_q\) of shape (n, 3).
        poni: torch.Tensor, optional
            The 6 ordered .poni calibration parameters as a tensor of shape (6,).
            Used only if ``uf_or_point`` is a point.
        indices : arraylike, optional
            The index of the \(u_q\) to consider. If not provided, all are considered.

        Returns
        -------
        material : torch.Tensor
            The int16 list of the predicted material index, shape (n',).
            The value -1, means that the spot is a fake spot.
        hkl : torch.Tensor
            The int16 list of the predicted class of 3 hkl, shape (n', 3).
            The value (0, 0, 0), means that the spot is a fake spot.
        confidence : torch.Tensor
            The quality of the prediction, interpretable as a percentage, shape (n').

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.nn import NNIndexator
        >>> lattice = torch.tensor([
        ...     [3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2],
        ... ])
        >>> indexator = NNIndexator(lattice)
        >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
        >>> point = torch.rand((1000, 2)) * 0.146
        >>> material, hkl, confidence = indexator.predict_hkl(point, poni=poni)
        >>> material.shape, hkl.shape, confidence.shape
        (torch.Size([1000]), torch.Size([1000, 3]), torch.Size([1000]))
        >>>
        """
        assert isinstance(uf_or_point, torch.Tensor), uf_or_point.__class__.__name__
        assert uf_or_point.ndim == 2, uf_or_point.shape

        # convert point to uf
        if uf_or_point.shape[-1] == 2:
            assert isinstance(poni, torch.Tensor), poni.__class__.__name__
            assert poni.shape == (6,), poni.shape
            uf_or_point = detector_to_ray(uf_or_point, poni)
        elif uf_or_point.shape[-1] == 3:
            if poni is not None:
                warnings.warn(
                    "the `poni` is ignored because uq has been provided, not point",
                    RuntimeWarning,
                )
        else:
            raise ValueError("only shape 2 and 3 allow")

        # convert uf to histogram
        u_q = uf_to_uq(uf_or_point)
        hist = self._uq_to_hist(u_q, indices=indices)

        # prediction
        pred = self.forward(hist)  # (n', nbr_families)

        # converting to a more interpretable space
        best = torch.argmax(pred, dim=1)  # (n')
        confidence = torch.take_along_dim(pred, best[:, None], dim=1).squeeze(1)
        hkl = self._families[best]
        material = torch.asarray(
            ([-1] + [i for i, f in enumerate(self._attrs["families"]) for _ in range(len(f))]),
            dtype=torch.int16,
            device=best.device,
        )[best]
        return material, hkl, confidence

    @property
    def weights(self) -> list[torch.Tensor]:
        """Return the trainaible neuronal network model weights."""
        return [
            param
            for layer in (self.layer_1, self.layer_2, self.layer_3, self.layer_4)
            for param in layer.parameters()
        ]
