#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import typing

import numpy as np
import torch

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.improc.spot.extrema import find_nb_extremums
from laueimproc.improc.spot.fit import fit_gaussian_em, fit_gaussians_em, fit_gaussians
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

    @auto_parallel
    def find_nb_extremums(self) -> torch.Tensor:
        """Find the number of extremums."""
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        return find_nb_extremums(self.rois)

    @auto_cache
    @auto_parallel
    def fit_gaussian_em(
        self,
        photon_density: typing.Union[torch.Tensor, np.ndarray, numbers.Real] = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by one gaussian using the EM algorithm in one shot, very fast.

        See ``laueimproc.gmm`` for terminology and ``laueimproc.gmm.em`` for the algo description.

        Parameters
        ----------
        photon_density : arraylike, optional
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussian_em``.
        **kwargs : dict
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussian_em``.

        Returns
        -------
        mean : torch.Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, 2). In the absolute diagram base.
        cov : torch.Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, 2, 2).
        infodict : dict[str]
            Comes from ``laueimproc.improc.spot.fit.fit_gaussian_em``.
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
    def fit_gaussians_em(
        self,
        photon_density: typing.Union[torch.Tensor, np.ndarray, numbers.Real] = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by \(K\) gaussians using the EM algorithm.

        See ``laueimproc.gmm`` for terminology and ``laueimproc.gmm.em`` for the algo description.

        Parameters
        ----------
        photon_density : arraylike, optional
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians_em``.
        **kwargs : dict
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians_em``.

        Returns
        -------
        mean : torch.Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2). In the absolute diagram base.
        cov : torch.Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
        eta : torch.Tensor
            The relative mass \(\eta\). Shape (n, \(K\)).
        infodict : dict[str]
            Comes from ``laueimproc.improc.spot.fit.fit_gaussians_em``.
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
        mean, cov, infodict = fit_gaussians_em(rois, photon_density, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift.unsqueeze(1)
        else:
            mean += shift.unsqueeze(1)

        # cast
        return mean, cov, infodict

    def fit_gaussian(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by one gaussian.

        Same as ``fit_gaussians`` but squeeze the \(K = 1\) dimension.
        """
        mean, cov, mass, infodict = self.fit_gaussians(*args, **kwargs, nbr_clusters=1)
        mean, cov, mass = mean.squeeze(1), cov.squeeze(1), mass.squeeze(1)
        if "eigtheta" in infodict:
            infodict = infodict.copy()  # to avoid insane cache reference error
            infodict["eigtheta"] = infodict["eigtheta"].squeeze(1)
        return mean, cov, mass, infodict

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
        mean : torch.Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2, 1). In the absolute diagram base.
        cov : torch.Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
        mass : torch.Tensor
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
