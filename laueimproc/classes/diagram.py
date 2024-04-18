#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import typing

import numpy as np
import torch

from laueimproc.improc.spot.basic import (
    compute_barycenters, compute_rois_max, compute_rois_sum
)
from laueimproc.improc.spot.extrema import find_nb_extremums
from laueimproc.improc.spot.fit import fit_gaussians_em, fit_gaussians
from laueimproc.improc.spot.rot_sym import compute_rot_sym
from laueimproc.opti.cache import auto_cache
from laueimproc.opti.parallel import auto_parallel
from .base_diagram import check_init, BaseDiagram



class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @auto_parallel  # automaticaly multithreading
    @check_init  # throws an exception if the diagram is not initialized
    def compute_barycenters(self, indexing: str = "ij") -> torch.Tensor:
        """Compute the barycenter of each spots."""
        barycenters = compute_barycenters(self.rois)  # relative to each spots
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)  # absolute

        assert isinstance(indexing, str), indexing.__class__.__name__
        assert indexing in {"ij", "xy"}, indexing
        if indexing == "xy":
            barycenters = torch.flip(barycenters, 1)
            barycenters += 0.5

        return barycenters

    @auto_parallel
    @check_init
    def compute_rois_max(self) -> torch.Tensor:
        """Get the intensity of the hottest pixel for each roi."""
        return compute_rois_max(self.rois)

    @auto_parallel
    @check_init
    def compute_rois_sum(self) -> torch.Tensor:
        """Sum the intensities of the pixels for each roi."""
        return compute_rois_sum(self.rois)

    @auto_cache
    @auto_parallel
    @check_init
    def compute_rot_sym(self) -> torch.Tensor:
        """Compute the similarity by rotation of each spots."""
        return compute_rot_sym(self.rois)

    @auto_cache
    @auto_parallel
    @check_init
    def compute_nb_extremums(self) -> torch.Tensor:
        """Find the number of extremums in each roi.

        Notes
        -----
        No noise filtering. Doesn't detect shoulders.
        """
        return find_nb_extremums(self.rois)

    def fit_gaussian_em(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by one gaussian using the EM algorithm in one shot, very fast.

        Same as ``flaueimproc.classes.diagram.fit_gaussians_em`` but squeeze the \(K = 1\) dim.

        Parameters
        ----------
        *args : tuple
            Transmitted to ``laueimproc.classes.diagram.fit_gaussians_em``.
        **kwargs : dict
            Transmitted to ``laueimproc.classes.diagram.fit_gaussians_em``.

        Returns
        -------
        mean : torch.Tensor
            The vectors \(\mathbf{\mu}\). Shape (n, 2). In the absolute diagram base.
        cov : torch.Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (n, 2, 2).
        infodict : dict[str]
            Comes from ``laueimproc.improc.spot.fit.fit_gaussian_em``.
        """
        assert "nbr_clusters" not in kwargs, "use fit_gaussianSSS_em instead"
        mean, cov, _, infodict = self.fit_gaussians_em(*args, **kwargs, nbr_clusters=1)
        mean, cov = mean.squeeze(1), cov.squeeze(1)
        if "eigtheta" in infodict:
            infodict["eigtheta"] = infodict["eigtheta"].squeeze(1)
        return mean, cov, infodict

    @auto_cache
    @auto_parallel
    @check_init
    def fit_gaussians_em(
        self,
        photon_density: typing.Union[torch.Tensor, np.ndarray, numbers.Real] = 1.0,
        indexing: str = "ij",
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
        mean, cov, eta, infodict = fit_gaussians_em(rois, photon_density, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift.unsqueeze(1)
        else:
            mean += shift.unsqueeze(1)

        assert isinstance(indexing, str), indexing.__class__.__name__
        assert indexing in {"ij", "xy"}, indexing
        if indexing == "xy":
            mean = torch.flip(mean, 2)
            mean += 0.5

        # cast
        return mean, cov, eta, infodict

    def fit_gaussian(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by one gaussian.

        Same as ``fit_gaussians`` but squeeze the \(K = 1\) dimension.
        """
        mean, cov, magnitude, infodict = self.fit_gaussians(*args, **kwargs, nbr_clusters=1)
        mean, cov, magnitude = mean.squeeze(1), cov.squeeze(1), magnitude.squeeze(1)
        if "eigtheta" in infodict:
            infodict = infodict.copy()  # to avoid insane cache reference error
            infodict["eigtheta"] = infodict["eigtheta"].squeeze(1)
        return mean, cov, magnitude, infodict

    @auto_cache
    @auto_parallel
    @check_init
    def fit_gaussians(
        self,
        loss: typing.Union[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str] = "mse",
        indexing: str = "ij",
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
        magnitude : torch.Tensor
            The absolute magnitude \(\theta.\eta\). Shape (n, \(K\)).
        infodict : dict[str]
            See ``laueimproc.improc.spot.fit.fit_gaussians``.
        """
        # preparation
        with self._rois_lock:
            data = self._rois[0]
            shapes = self._rois[1][:, 2:]
            shift = self._rois[1][:, :2]

        # main fit
        mean, cov, magnitude, infodict = fit_gaussians(data, shapes, loss, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift.reshape(-1, 1, 2, 1)
        else:
            mean += shift.reshape(-1, 1, 2, 1)

        assert isinstance(indexing, str), indexing.__class__.__name__
        assert indexing in {"ij", "xy"}, indexing
        if indexing == "xy":
            mean = torch.flip(mean, 2)
            mean += 0.5

        return mean, cov, magnitude, infodict
