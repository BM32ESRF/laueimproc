#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import typing

import numpy as np
import torch

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.improc.spot.fit import fit_gaussian_em, fit_gaussians
from laueimproc.improc.spot.rot_sym import compute_rot_sym
from laueimproc.opti.cache import auto_cache
from laueimproc.opti.parallel import auto_parallel
from .base_diagram import BaseDiagram


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @auto_parallel  # automaticaly multithreading
    def compute_barycenters(self) -> torch.Tensor:
        """Compute the barycenter of each spots."""
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        barycenters = compute_barycenters(self.rois)  # relative to each spots
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)  # absolute
        return barycenters

    @auto_parallel
    def compute_pxl_intensities(self) -> torch.Tensor:
        """Compute the total pixel intensity for each spots."""
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        return compute_pxl_intensities(self.rois)

    @auto_cache
    @auto_parallel
    def compute_rot_sym(self) -> torch.Tensor:
        """Compute the similarity by rotation of each spots."""
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        return compute_rot_sym(self.rois)

    @auto_cache
    @auto_parallel
    def fit_gaussian_em(
        self,
        photon_density: typing.Union[torch.Tensor, np.ndarray, numbers.Real] = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by one gaussian.

        See ``laueimproc.improc.gmm`` for terminology.

        Parameters
        ----------
        photon_density : arraylike, optional
            See to ``laueimproc.improc.spot.fit.fit_gaussian_em``.
        **kwargs : dict
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussian_em``.

        Returns
        -------
        mean : Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, 2). In the absolute diagram base.
        cov : Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, 2, 2).
        infodict : dict[str]
            A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
        """
        # preparation
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        photon_density = (
            float(photon_density)
            if isinstance(photon_density, numbers.Real)
            else torch.as_tensor(photon_density, dtype=torch.float32)
        )
        rois = self.rois
        shift = self.bboxes[:, :2]

        # main fit
        mean, cov, infodict = fit_gaussian_em(rois, photon_density, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift
        else:
            mean += shift

        # cast
        return mean, cov, infodict

    @auto_cache
    @auto_parallel
    def fit_gaussians(
        self,
        loss: typing.Union[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str] = "mse",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by \(K\) gaussians.

        See ``laueimproc.improc.gmm`` for terminology.

        Parameters
        ----------
        loss : callable or str, default="mse"
            Quantify the difference between ``self.rois`` and estimated rois from the gmm.
            The specific string values are understood:

            * "l1" (absolute difference): \(
                \frac{1}{H.W}
                \sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1}
                | rois_{k,i,j} - rois\_pred_{k,i,j} |
            \)

            * "mse" (mean square error): \(
                \frac{1}{H.W}
                \sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1}
                \left( rois_{k,i,j} - rois\_pred_{k,i,j} \right)^2
            \)

        **kwargs : dict
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians``.

        Returns
        -------
        mean : Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2, 1). In the absolute diagram base.
        cov : Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
        mass : Tensor
            The absolute mass \(\theta.\eta\). Shape (n, \(K\)).
        infodict : dict[str]
            See ``laueimproc.improc.spot.fit.fit_gaussians``.
        """
        # preparation
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        if isinstance(loss, str):
            assert loss in {"l1", "mse"}, loss
            loss = {
                "l1": lambda rois, rois_pred: torch.mean(torch.abs(rois-rois_pred), dim=(-1, -2)),
                "mse": lambda rois, rois_pred: torch.mean((rois-rois_pred)**2, dim=(-1, -2)),
            }[loss]
        rois = self.rois
        shift = self.bboxes[:, :2]

        # main fit
        mean, cov, mass, infodict = fit_gaussians(rois, loss, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift.reshape(-1, 1, 2, 1)
        else:
            mean += shift.reshape(-1, 1, 2, 1)

        return mean, cov, mass, infodict
