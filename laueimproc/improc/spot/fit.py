#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import numbers
import typing

import torch

try:
    from laueimproc.gmm import c_fit
except ImportError:
    logging.warning(
        "failed to import laueimproc.gmm.em.c_em, a slow python version is used instead"
    )
    c_fit = None
from laueimproc.opti.rois import rawshapes2rois


def _apply_to_metric(
    meytric: typing.Callable,
    data: bytearray,
    bboxes: torch.Tensor,
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    rois = rawshapes2rois(data, bboxes[:, 2:])
    rois = [rois[i, :h, :w] for i, (h, w) in enumerate(bboxes[:, 2:].tolist())]
    out = []
    for i, roi in enumerate(rois):
        points_i, points_j = torch.meshgrid(
            torch.arange(0.5, roi.shape[0]+0.5, dtype=roi.dtype, device=roi.device),
            torch.arange(0.5, roi.shape[1]+0.5, dtype=roi.dtype, device=roi.device),
            indexing="ij",
        )
        points_i, points_j = points_i.ravel(), points_j.ravel()
        obs = torch.cat([points_i.unsqueeze(1), points_j.unsqueeze(1)], dim=1)  # (n_obs, n_var)
        weights = roi.ravel()
        out.append(
            torch.asarray(meytric(obs, weights, (gmm[0][i], gmm[1][i], gmm[2][i]))).unsqueeze(0)
        )
    return torch.cat(out)


def fit_gaussians_em(
    data: bytearray, bboxes: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.
    nbr_clusters, nbr_tries : int
        Transmitted to ``laueimproc.gmm.fit.fit_em``.
    aic, bic, log_likelihood : boolean, default=False
        If set to True, the metric is happend into `infodict`.
        The metrics are computed in the ``laueimproc.gmm.metric`` module.

        * aic: Akaike Information Criterion of shape (...,). \(aic = 2p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the likelihood.
        * bic: Bayesian Information Criterion of shape (...,).
        \(bic = \log(N)p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the likelihood.
        * log_likelihood of shape (...,): \(
            \log\left(L_{\alpha,\omega}\right) = \log\left(
                \prod\limits_{i=1}^N \sum\limits_{j=1}^K
                \eta_j \left(
                    \mathcal{N}_{(\mathbf{\mu}_j,\frac{1}{\omega_i}\mathbf{\Sigma}_j)}(\mathbf{x}_i)
                \right)^{\alpha_i}
            \right)
        \)

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
    >>> import torch
    >>> from laueimproc.gmm.linalg import multivariate_normal
    >>> from laueimproc.improc.spot.fit import fit_gaussians_em
    >>>
    >>> cov_ref = torch.asarray([[[2, 1], [1, 4]], [[3, -3], [-3, 9]], [[4, 0], [0, 4]]])
    >>> cov_ref = cov_ref.to(torch.float32)
    >>> mean_ref = torch.asarray([[[10.0, 15.0], [20.0, 15.0], [15.0, 20.0]]])
    >>> obs = multivariate_normal(cov_ref, 3_000_000) + mean_ref
    >>> obs = obs.to(torch.int32).reshape(-1, 2)
    >>> height, width = 35, 35
    >>> rois = obs[:, 0]*width + obs[:, 1]
    >>> rois = rois[torch.logical_and(rois >= 0, rois < height * width)]
    >>> rois = torch.bincount(rois, minlength=height*width).to(torch.float32)
    >>> rois = rois.reshape(1, height, width) / rois.max()
    >>>
    >>> data = bytearray(rois.to(torch.float32).numpy().tobytes())
    >>> bboxes = torch.asarray([[0, 0, height, width]], dtype=torch.int16)
    >>> mean, cov, eta, infodict = fit_gaussians_em(
    ...     data, bboxes, nbr_clusters=3, aic=True, bic=True, log_likelihood=True
    ... )
    >>> any(torch.allclose(mu, mean_ref[0, 0], atol=0.5) for mu in mean[0])
    True
    >>> any(torch.allclose(mu, mean_ref[0, 1], atol=0.5) for mu in mean[0])
    True
    >>> any(torch.allclose(mu, mean_ref[0, 2], atol=0.5) for mu in mean[0])
    True
    >>> any(torch.allclose(c, cov_ref[0], atol=1.0) for c in cov[0])
    True
    >>> any(torch.allclose(c, cov_ref[1], atol=1.0) for c in cov[0])
    True
    >>> any(torch.allclose(c, cov_ref[2], atol=1.0) for c in cov[0])
    True
    >>>
    >>> # import matplotlib.pyplot as plt
    >>> # _ = plt.imshow(rois[0], extent=(0, width, height, 0))
    >>> # _ = plt.scatter(mean[..., 1].ravel(), mean[..., 0].ravel())
    >>> # plt.show()
    >>>
    """
    # compute gmm
    if not kwargs.get("_no_c", False) and c_fit is not None:
        mean, cov, eta = c_fit.fit_em(
            data, bboxes.numpy(force=True),
            nbr_clusters=kwargs.get("nbr_clusters", 3),
            nbr_tries=kwargs.get("nbr_tries", 4),
        )
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(cov)
        eta = torch.from_numpy(eta)
    else:
        from laueimproc.gmm import fit
        rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=kwargs.get("_no_c", False))
        rois = [rois[i, :h, :w] for i, (h, w) in enumerate(bboxes[:, 2:].tolist())]
        mean, cov, eta = [], [], []
        for roi in rois:
            mean_, cov_, eta_ = fit.fit_em(
                roi,
                nbr_clusters=kwargs.get("nbr_clusters", 3),
                nbr_tries=kwargs.get("nbr_tries", 4),
            )
            mean.append(mean_.unsqueeze(0))
            cov.append(cov_.unsqueeze(0))
            eta.append(eta_.unsqueeze(0))
        if len(bboxes) == 0:
            nbr_clusters = kwargs.get("nbr_clusters", 1)
            assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
            assert nbr_clusters > 0, nbr_clusters
            mean = torch.empty((0, nbr_clusters, 2), dtype=torch.float32)
            cov = torch.empty((0, nbr_clusters, 2, 2), dtype=torch.float32)
            eta = torch.empty((0, nbr_clusters), dtype=torch.float32)
        else:
            mean = torch.cat(mean)
            cov = torch.cat(cov)
            eta = torch.cat(eta)
        mean += bboxes[:, :2].unsqueeze(1)  # relative to absolute

    # cast output
    mean = mean.to(bboxes.device)
    cov = cov.to(bboxes.device)
    eta = eta.to(bboxes.device)

    # fill infodict
    infodict = {}
    if kwargs.get("aic", False):
        from laueimproc.gmm import metric
        infodict["aic"] = _apply_to_metric(metric.aic, data, bboxes, (mean, cov, eta))
    if kwargs.get("bic", False):
        from laueimproc.gmm import metric
        infodict["bic"] = _apply_to_metric(metric.bic, data, bboxes, (mean, cov, eta))
    if kwargs.get("log_likelihood", False):
        from laueimproc.gmm import metric
        infodict["log_likelihood"] = (
            _apply_to_metric(metric.log_likelihood, data, bboxes, (mean, cov, eta))
        )

    return mean, cov, eta, infodict


def fit_gaussians_mse(
    data: bytearray, bboxes: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians by minimising the mean square error.

    See ``laueimproc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.
    nbr_clusters, nbr_tries : int
        Transmitted to ``laueimproc.gmm.fit.fit_mse``.
    mse : boolean, default=False
        If set to True, the metric is happend into `infodict`.
        The metrics are computed in the ``laueimproc.gmm.metric`` module.

        * mse: Mean Square Error of shape (...,). \(
            mse = \frac{1}{N}\sum\limits_{i=1}^N
                \left(\left(\sum\limits_{j=1}^K
                \eta_j \left(
                    \mathcal{N}_{(\mathbf{\mu}_j,\frac{1}{\omega_i}\mathbf{\Sigma}_j)}(\mathbf{x}_i)
                \right)\right) - \alpha_i\right)^2
        \)

    Returns
    -------
    mean : torch.Tensor
        The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2). In the absolute diagram base.
    cov : torch.Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
    mag : torch.Tensor
        The absolute magnitude \(\eta\). Shape (n, \(K\)).
    infodict : dict[str]
        A dictionary of optional outputs.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.linalg import multivariate_normal
    >>> from laueimproc.improc.spot.fit import fit_gaussians_mse
    >>>
    >>> cov_ref = torch.asarray([[[2, 1], [1, 4]], [[3, -3], [-3, 9]]])
    >>> cov_ref = cov_ref.to(torch.float32)
    >>> mean_ref = torch.asarray([[[10.0, 15.0], [20.0, 15.0]]])
    >>> obs = multivariate_normal(cov_ref, 2_000_000) + mean_ref
    >>> obs = obs.to(torch.int32).reshape(-1, 2)
    >>> height, width = 30, 30
    >>> rois = obs[:, 0]*width + obs[:, 1]
    >>> rois = rois[torch.logical_and(rois >= 0, rois < height * width)]
    >>> rois = torch.bincount(rois, minlength=height*width).to(torch.float32)
    >>> rois = rois.reshape(1, height, width) / rois.max()
    >>>
    >>> data = bytearray(rois.to(torch.float32).numpy().tobytes())
    >>> bboxes = torch.asarray([[0, 0, height, width]], dtype=torch.int16)
    >>> mean, cov, mag, _ = fit_gaussians_mse(data, bboxes, nbr_clusters=2)
    >>> any(torch.allclose(mu, mean_ref[0, 0], atol=0.5) for mu in mean[0])
    True
    >>> any(torch.allclose(mu, mean_ref[0, 1], atol=0.5) for mu in mean[0])
    True
    >>>
    >>> # import matplotlib.pyplot as plt
    >>> # _ = plt.imshow(rois[0], extent=(0, width, height, 0))
    >>> # _ = plt.scatter(mean[..., 1].ravel(), mean[..., 0].ravel())
    >>> # plt.show()
    >>>
    """
    # compute gmm
    if not kwargs.get("_no_c", False) and c_fit is not None:
        mean, cov, mag = c_fit.fit_mse(
            data, bboxes.numpy(force=True),
            nbr_clusters=kwargs.get("nbr_clusters", 3),
            nbr_tries=kwargs.get("nbr_tries", 4),
        )
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(cov)
        mag = torch.from_numpy(mag)
    else:
        from laueimproc.gmm import fit
        rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=kwargs.get("_no_c", False))
        rois = [rois[i, :h, :w] for i, (h, w) in enumerate(bboxes[:, 2:].tolist())]
        mean, cov, mag = [], [], []
        for roi in rois:
            mean_, cov_, mag_ = fit.fit_mse(
                roi,
                nbr_clusters=kwargs.get("nbr_clusters", 3),
                nbr_tries=kwargs.get("nbr_tries", 4),
            )
            mean.append(mean_.unsqueeze(0))
            cov.append(cov_.unsqueeze(0))
            mag.append(mag_.unsqueeze(0))
        if len(bboxes) == 0:
            nbr_clusters = kwargs.get("nbr_clusters", 1)
            assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
            assert nbr_clusters > 0, nbr_clusters
            mean = torch.empty((0, nbr_clusters, 2), dtype=torch.float32)
            cov = torch.empty((0, nbr_clusters, 2, 2), dtype=torch.float32)
            mag = torch.empty((0, nbr_clusters), dtype=torch.float32)
        else:
            mean = torch.cat(mean)
            cov = torch.cat(cov)
            mag = torch.cat(mag)
        mean += bboxes[:, :2].unsqueeze(1)  # relative to absolute

    # cast output
    mean = mean.to(bboxes.device)
    cov = cov.to(bboxes.device)
    mag = mag.to(bboxes.device)

    # fill infodict
    infodict = {}
    if kwargs.get("mse", False):
        from laueimproc.gmm import metric
        infodict["mse"] = _apply_to_metric(metric.mse, data, bboxes, (mean, cov, mag))

    return mean, cov, mag, infodict
