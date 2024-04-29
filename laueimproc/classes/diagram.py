#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import typing

import numpy as np
import torch

from laueimproc.improc.spot.fit import fit_gaussians_em, fit_gaussians
from laueimproc.opti.cache import auto_cache
from .base_diagram import check_init, BaseDiagram


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @check_init  # throws an exception if the diagram is not initialized
    def compute_rois_centroid(self, **_kwargs) -> torch.Tensor:
        """Compute the barycenter of each spots.

        Returns
        -------
        positions : torch.Tensor
            The 2 barycenter position for each roi.
            Each line corresponds to a spot and each column to an axis (shape (n, 2)).
            See ``laueimproc.improc.spot.basic.compute_rois_centroid`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_centroid())
        tensor([[1987.1576,  896.4261],
                [1945.4674,  913.8295],
                [1908.5780,  971.4541],
                ...,
                [  55.2341, 1352.7832],
                [  19.2854, 1208.5648],
                [   9.2786,  904.4847]])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_centroid
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_centroid(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def compute_rois_max(self, **_kwargs) -> torch.Tensor:
        """Get the intensity and the position of the hottest pixel for each roi.

        Returns
        -------
        imax_pos1_pos2 : torch.Tensor
            The concatenation of the colum vectors of the argmax and the intensity (shape (n, 3)).
            See ``laueimproc.improc.spot.basic.compute_rois_max`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_max())
        tensor([[1.9875e+03, 8.9650e+02, 4.1199e-04],
                [1.9455e+03, 9.1350e+02, 7.9698e-02],
                [1.9085e+03, 9.7150e+02, 4.1199e-04],
                ...,
                [5.4500e+01, 1.3535e+03, 5.4932e-04],
                [1.8500e+01, 1.2085e+03, 4.5777e-04],
                [8.5000e+00, 9.0450e+02, 4.4251e-04]])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_max
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_max(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def compute_rois_nb_peaks(self, **_kwargs) -> torch.Tensor:
        """Find the number of extremums in each roi.

        Returns
        -------
        nbr_of_peaks : torch.Tensor
            The number of extremums (shape (n,)).
            See ``laueimproc.improc.spot.extrema.compute_rois_nb_peaks`` for more details.

        Notes
        -----
        No noise filtering. Doesn't detect shoulders.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_nb_peaks())  # doctest: +ELLIPSIS
        tensor([ 5,  6,  1,  2,  2,  3,  2,  3,  5,  1,  2,  2,  4,  3,  5,  4,  2,  2,
                 2,  3,  2,  4,  4,  3,  2,  2,  4,  7,  2,  1,  3,  4,  1,  5,  4,  1,
                 5,  2,  3,  3,  3,  5,  3,  1,  4,  4,  3,  3,  7,  2,  5,  2,  4,  3,
                 ...
                 1,  3,  3,  2,  4,  1,  2,  1,  2,  3,  2,  4,  4,  4,  2,  3,  1,  3,
                 4,  3,  1,  5,  2,  4,  7,  2,  2,  2,  4,  2,  2,  3,  2,  4,  2,  4,
                 3,  2,  5,  2,  3,  2,  3,  1,  2,  2,  1,  1,  2,  3,  3,  5,  4,  4,
                 3,  2,  3,  2,  3,  2,  1], dtype=torch.int16)
        >>>
        """
        from laueimproc.improc.spot.extrema import compute_rois_nb_peaks
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_nb_peaks(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def compute_rois_pca(self, **_kwargs) -> torch.Tensor:
        """Compute the pca on each spot.

        Returns
        -------
        std1_std2_theta : torch.Tensor
            The concatenation of the colum vectors of the two std and the angle (shape (n, 3)).
            See ``laueimproc.improc.spot.pca.compute_rois_pca`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_pca())
        tensor([[ 0.2694,  0.2636, -1.4625],
                [ 0.1830,  0.1499,  0.0815],
                [ 0.2784,  0.2402,  0.3340],
                ...,
                [ 0.3560,  0.2605,  1.1708],
                [ 0.3535,  0.1901, -1.3840],
                [ 0.3169,  0.2077,  1.4719]])
        >>>
        """
        from laueimproc.improc.spot.pca import compute_rois_pca
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_pca(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def compute_rois_sum(self, **_kwargs) -> torch.Tensor:
        """Sum the intensities of the pixels for each roi.

        Returns
        -------
        total_intensity : torch.Tensor
            The intensity of each roi, sum of the pixels (shape (n,)).
            See ``laueimproc.improc.spot.basic.compute_rois_sum`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_sum())  # doctest: +ELLIPSIS
        tensor([3.9216e-03, 5.7542e-01, 6.6529e-03, 1.6938e-03, 1.0986e-02, 2.4811e-02,
                2.0935e-02, 3.5966e-02, 2.7893e-02, 4.9592e-03, 5.6901e-02, 1.0986e-03,
                1.0588e-01, 1.9989e-03, 4.2100e-02, 4.5542e-01, 8.2399e-03, 3.0842e-01,
                ...
                5.3590e-02, 1.7853e-03, 4.1199e-03, 9.8131e-02, 2.2431e-03, 1.8158e-03,
                2.8077e-03, 6.2394e-02, 2.5025e-03, 2.3697e-02, 2.5284e-02, 5.3864e-03,
                1.3931e-02, 9.8726e-03, 3.4943e-03, 2.1515e-03, 2.6398e-03, 3.7690e-03,
                3.9979e-03])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_sum
        self.flush()
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_sum(data, bboxes, **_kwargs)

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
