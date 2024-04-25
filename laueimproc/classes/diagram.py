#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import typing

import numpy as np
import torch

from laueimproc.improc.spot.basic import compute_rois_max, compute_rois_sum
from laueimproc.improc.spot.extrema import find_nb_extremums
from laueimproc.improc.spot.fit import fit_gaussians_em, fit_gaussians
from laueimproc.improc.spot.pca import pca
from laueimproc.opti.cache import auto_cache
from .base_diagram import check_init, BaseDiagram


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @check_init  # throws an exception if the diagram is not initialized
    def compute_barycenters(self) -> torch.Tensor:
        """Compute the barycenter of each spots.

        Returns
        -------
        positions : torch.Tensor
            The 2 barycenter position for each roi.
            Each line corresponds to a spot and each column to an axis (shape (n, 2)).

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_barycenters())
        tensor([[1987.1576,  896.4261],
                [1945.4674,  913.8295],
                [1908.5780,  971.4541],
                ...,
                [  55.2341, 1352.7832],
                [  19.2854, 1208.5648],
                [   9.2786,  904.4847]])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_barycenters
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_barycenters(data, bboxes)

    @auto_cache
    @check_init
    def compute_pca(self) -> torch.Tensor:
        """Compute the pca on each spot.

        Returns
        -------
        std1_std2_theta : torch.Tensor
            Return the same values as ``laueimproc.improc.spot.pca``.
        """
        data, shapes = self._rois[0], self._rois[1][:, 2:].numpy(force=True)
        return pca(data, shapes)

    @check_init
    def compute_rois_max(self) -> torch.Tensor:
        """Get the intensity of the hottest pixel for each roi."""
        return compute_rois_max(self.rois)

    @check_init
    def compute_rois_sum(self) -> torch.Tensor:
        """Sum the intensities of the pixels for each roi."""
        return compute_rois_sum(self.rois)

    @auto_cache
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
        indexing : str, default="ij"
            The convention used for the returned positions values. Can be "ij" or "xy".
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
            else torch.asarray(photon_density, dtype=torch.float32)
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
    @check_init
    def fit_gaussians(
        self, loss: str = "mse", indexing: str = "ij", **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by \(K\) gaussians.

        See ``laueimproc.improc.gmm`` for terminology.

        Parameters
        ----------
        loss : str, default="mse"
            Quantify the difference between ``self.rois`` and estimated rois from the gmm.
            The possible values are:

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
        indexing : str, default="ij"
            The convention used for the returned positions values. Can be "ij" or "xy".
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
        assert loss == "mse", "only mse is implemented"

        # preparation
        with self._rois_lock:
            data = self._rois[0]
            shapes = self._rois[1][:, 2:]
            shift = self._rois[1][:, :2]

        # main fit
        mean, cov, magnitude, infodict = fit_gaussians(data, shapes, **kwargs)

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
