#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import torch

from laueimproc.convention import ij_to_xy_decorator
from laueimproc.opti.cache import auto_cache
from .base_diagram import check_init, BaseDiagram


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @ij_to_xy_decorator(i=(slice(None), 0), j=(slice(None), 1))  # append conv argument
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
        >>> print(diagram.compute_rois_centroid())  # doctset: +ELLIPSIS
        tensor([[4.2637e+00, 5.0494e+00],
                [7.2805e-01, 2.6091e+01],
                [5.1465e-01, 1.9546e+03],
                ...,
                [1.9119e+03, 1.8759e+02],
                [1.9376e+03, 1.9745e+03],
                [1.9756e+03, 1.1794e+03]])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_centroid
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_centroid(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def compute_rois_inertia(self, **_kwargs) -> torch.Tensor:
        """Compute the inertia of the spot along the center of mass.

        Returns
        -------
        inertia : torch.Tensor
            The order 2 momentum centered along the barycenter (shape (n,)).
            See ``laueimproc.improc.spot.basic.compute_rois_inertia`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_inertia())  # doctset: +ELLIPSIS
        tensor([4.0506e+00, 9.7974e-02, 1.6362e-01, 2.1423e-03, 4.8515e-02, 2.6657e-01,
                9.9830e-02, 1.7221e-03, 2.8771e-02, 5.3754e-03, 2.4759e-02, 8.7665e-01,
                5.1118e-02, 9.6223e-02, 3.0648e-01, 1.1362e-01, 2.6335e-01, 1.1388e-01,
                ...,
                4.8586e+00, 4.4373e+00, 1.3409e-01, 3.1009e-01, 1.5538e-02, 8.4468e-03,
                1.5235e+00, 1.0291e+01, 8.1125e+00, 5.4182e-01, 1.4235e+00, 4.4005e+00,
                7.0188e+00, 2.5770e+00, 1.1837e+01, 7.0610e+00, 3.6725e+00, 2.2335e+01])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_inertia
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_inertia(data, bboxes, **_kwargs)

    @auto_cache
    @ij_to_xy_decorator(i=(slice(None), 1), j=(slice(None), 2))
    @check_init
    def compute_rois_max(self, **_kwargs) -> torch.Tensor:
        """Get the intensity and the position of the hottest pixel for each roi.

        Returns
        -------
        pos1_pos2_imax : torch.Tensor
            The concatenation of the colum vectors of the argmax and the intensity (shape (n, 3)).
            See ``laueimproc.improc.spot.basic.compute_rois_max`` for more details.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> print(diagram.compute_rois_max())  # doctset: +ELLIPSIS
        tensor([[2.5000e+00, 5.5000e+00, 9.2165e-03],
                [5.0000e-01, 2.2500e+01, 1.3581e-03],
                [5.0000e-01, 1.9545e+03, 1.3123e-02],
                ...,
                [1.9125e+03, 1.8750e+02, 1.1250e-01],
                [1.9375e+03, 1.9745e+03, 6.0273e-02],
                [1.9755e+03, 1.1795e+03, 2.9212e-01]])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_max
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
        >>> print(diagram.compute_rois_nb_peaks())
        tensor([23,  4,  3,  2,  6,  3,  1,  3,  1,  4,  1,  2,  3,  1,  1,  3,  3,  3,
                 2,  7,  4,  1,  1,  3,  3,  2,  4,  2,  1,  2,  1,  2,  2,  2,  2,  2,
                 6,  2,  1,  2,  4,  2,  3,  4,  1,  1,  4,  3,  1,  4,  4,  5,  3,  3,
                 1,  5,  3,  2,  6,  1,  2,  3,  1,  4,  6,  3,  6,  2,  2,  5,  7,  5,
                 3,  2,  3,  1,  2,  6,  8,  2,  3,  3,  3,  3,  2,  4,  2,  1,  1,  4,
                 4,  1,  4,  2,  5,  4,  3,  3,  1,  2,  1,  3,  4,  2,  5,  5,  7,  2,
                 6,  3,  5,  2,  2,  2,  7,  2,  3,  5,  3,  1,  4,  8,  3,  3,  4,  3,
                 4,  2,  5,  1,  7,  3,  5,  4, 10,  2,  3,  3,  4,  3,  5,  8,  2,  5,
                 8,  3,  2,  3,  5,  5,  5,  3,  4,  6,  2,  4,  1,  3,  4,  2,  3,  4,
                 5,  5,  2,  4,  6,  4,  4,  1,  2,  4,  4,  3, 13, 13,  3,  6,  3,  2,
                 1,  4,  6,  3,  2,  2, 16,  3,  1,  2,  2,  4,  1,  3,  7,  4,  1,  5,
                 4,  7,  5,  1,  6,  6,  2,  4,  3,  9,  3,  2,  3,  2,  8,  5,  3,  3,
                 3,  3, 10,  9,  2,  2,  5,  4,  3,  3,  2,  1,  3,  6, 11,  2,  4,  4,
                 6,  2,  7,  5,  2, 10], dtype=torch.int16)
        >>>
        """
        from laueimproc.improc.spot.extrema import compute_rois_nb_peaks
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
        >>> print(diagram.compute_rois_pca())  # doctset: +ELLIPSIS
        tensor([[ 3.5272e+00,  2.4630e+00, -8.8861e-01],
                [ 2.8604e+00,  4.8286e-01, -1.5335e+00],
                [ 1.6986e+00,  1.5142e-01, -1.5628e+00],
                ...,
                [ 2.1297e+00,  1.7486e+00, -2.6128e-01],
                [ 2.0484e+00,  1.7913e+00,  4.9089e-01],
                [ 2.6780e+00,  1.9457e+00,  9.9400e-02]])
        >>>
        """
        from laueimproc.improc.spot.pca import compute_rois_pca
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
        >>> print(diagram.compute_rois_sum())  # doctset: +ELLIPSIS
        tensor([2.1886e-01, 1.1643e-02, 5.6260e-02, 1.6938e-03, 9.8726e-03, 6.1647e-02,
                3.5279e-02, 3.1891e-03, 1.0071e-02, 2.2889e-03, 1.0285e-02, 1.9760e-01,
                1.7258e-02, 3.3402e-02, 9.1829e-02, 3.1510e-02, 8.4550e-02, 4.0864e-02,
                ...,
                9.2822e-01, 9.7104e-01, 4.1733e-02, 9.4377e-02, 5.2491e-03, 4.2115e-03,
                3.9217e-01, 1.5907e+00, 1.1802e+00, 1.4968e-01, 4.0696e-01, 6.3442e-01,
                1.3559e+00, 6.0548e-01, 1.7116e+00, 9.2990e-01, 4.9596e-01, 2.0383e+00])
        >>>
        """
        from laueimproc.improc.spot.basic import compute_rois_sum
        with self._rois_lock:
            data, bboxes = self._rois
        return compute_rois_sum(data, bboxes, **_kwargs)

    @auto_cache
    @check_init
    def fit_gaussians_em(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        r"""Fit each roi by \(K\) gaussians using the EM algorithm.

        See ``laueimproc.gmm`` for terminology and ``laueimproc.gmm.em`` for the algo description.

        Parameters
        ----------
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
            A dictionary of optional outputs.

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_sample
        >>> diagram = Diagram(get_sample())
        >>> diagram.find_spots()
        >>> mean, cov, eta, _ = diagram.fit_gaussians_em(nbr_clusters=3, nbr_tries=4)
        >>> mean.shape, cov.shape, eta.shape
        (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3]))
        >>>
        """
        from laueimproc.improc.spot.fit import fit_gaussians_em
        with self._rois_lock:
            data, bboxes = self._rois
        return fit_gaussians_em(data, bboxes, **kwargs)

    # def fit_gaussian(
    #     self, *args, **kwargs
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    #     r"""Fit each roi by one gaussian.

    #     Same as ``fit_gaussians`` but squeeze the \(K = 1\) dimension.
    #     """
    #     mean, cov, magnitude, infodict = self.fit_gaussians(*args, **kwargs, nbr_clusters=1)
    #     mean, cov, magnitude = mean.squeeze(1), cov.squeeze(1), magnitude.squeeze(1)
    #     if "eigtheta" in infodict:
    #         infodict = infodict.copy()  # to avoid insane cache reference error
    #         infodict["eigtheta"] = infodict["eigtheta"].squeeze(1)
    #     return mean, cov, magnitude, infodict

    # @auto_cache
    # @check_init
    # def fit_gaussians(
    #     self, loss: str = "mse", indexing: str = "ij", **kwargs
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    #     r"""Fit each roi by \(K\) gaussians.

    #     See ``laueimproc.gmm`` for terminology.

    #     Parameters
    #     ----------
    #     loss : str, default="mse"
    #         Quantify the difference between ``self.rois`` and estimated rois from the gmm.
    #         The possible values are:

    #         * "l1" (absolute difference): \(
    #             \frac{1}{H.W}
    #             \sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1}
    #             | rois_{k,i,j} - rois\_pred_{k,i,j} |
    #         \)

    #         * "mse" (mean square error): \(
    #             \frac{1}{H.W}
    #             \sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1}
    #             \left( rois_{k,i,j} - rois\_pred_{k,i,j} \right)^2
    #         \)
    #     indexing : str, default="ij"
    #         The convention used for the returned positions values. Can be "ij" or "xy".
    #     **kwargs : dict
    #         Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians``.

    #     Returns
    #     -------
    #     mean : torch.Tensor
    #         The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2, 1). In the absolute diagram base.
    #     cov : torch.Tensor
    #         The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
    #     magnitude : torch.Tensor
    #         The absolute magnitude \(\theta.\eta\). Shape (n, \(K\)).
    #     infodict : dict[str]
    #         See ``laueimproc.improc.spot.fit.fit_gaussians``.
    #     """
    #     assert loss == "mse", "only mse is implemented"

    #     # preparation
    #     with self._rois_lock:
    #         data = self._rois[0]
    #         shapes = self._rois[1][:, 2:]
    #         shift = self._rois[1][:, :2]

    #     # main fit
    #     mean, cov, magnitude, infodict = fit_gaussians(data, shapes, **kwargs)

    #     # spot base to diagram base
    #     if mean.requires_grad:
    #         mean = mean + shift.reshape(-1, 1, 2, 1)
    #     else:
    #         mean += shift.reshape(-1, 1, 2, 1)

    #     assert isinstance(indexing, str), indexing.__class__.__name__
    #     assert indexing in {"ij", "xy"}, indexing
    #     if indexing == "xy":
    #         mean = torch.flip(mean, 2)
    #         mean += 0.5

    #     return mean, cov, magnitude, infodict
