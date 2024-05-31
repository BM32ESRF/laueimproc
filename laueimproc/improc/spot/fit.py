#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import numbers

import numpy as np
import torch

try:
    from laueimproc.gmm.em import c_em
except ImportError:
    logging.warning(
        "failed to import laueimproc.gmm.em.c_em, a slow python version is used instead"
    )
    c_em = None
    from laueimproc.gmm.em import em
else:
    em = None
from laueimproc.gmm.gmm import cost_and_grad
from laueimproc.opti.rois import rawshapes2rois


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
    **dict : dict
        Transmitted to ``laueimproc.gmm.em``.

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
    """
    if not kwargs.get("_no_c", False) and c_em is not None:
        mean, cov, eta = c_em(data, bboxes.numpy(force=True), **kwargs)
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(cov)
        eta = torch.from_numpy(eta)
    else:
        rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=kwargs.get("_no_c", False))
        rois = [rois[i, :h, :w] for i, (h, w) in enumerate(bboxes[:, 2:].tolist())]
        mean, cov, eta = [], [], []
        for roi in rois:
            mean_, cov_, eta_ = em(roi, **kwargs)
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

    mean = mean.to(bboxes.device)
    cov = cov.to(bboxes.device)
    eta = eta.to(bboxes.device)
    infodict = {}
    return mean, cov, eta, infodict


def fit_gaussians(
    data: bytearray, shapes: np.ndarray[np.int32], **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
    shapes : np.ndarray[np.int32]
        Contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.
    **kwargs : dict
        Transmitted to ``laueimproc.gmm.em``, used for initialisation.

    Returns
    -------
    mean : torch.Tensor
        The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2, 1). In the relative rois base.
    cov : torch.Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
    mass : torch.Tensor
        The absolute mass \(\theta.\eta\). Shape (n, \(K\)).
    infodict : dict[str]

        * "loss" : torch.Tensor
            The vector of loss. Shape (n,).
        * "pred" : torch.Tensor
            The predicted rois tensor. Shape(n, h, w)
    """
    # verification
    assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__

    rois = rawshapes2rois(data, shapes.numpy(force=True))

    # initial guess
    obs = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    obs = (obs[0].ravel(), obs[1].ravel())
    obs = torch.cat([obs[0].unsqueeze(-1), obs[1].unsqueeze(-1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    mean, cov, mag = em(
        obs, torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2])), **kwargs
    )
    infodict = {}
    mag *= torch.amax(rois, dim=(-1, -2))
    init = (mean.clone(), cov.clone(), mag.clone())

    # overfit
    for i, (height, width) in enumerate(shapes.tolist()):
        mean[i], cov[i], mag[i] = _fit_gaussians_no_batch(
            rois[i, :height, :width], mean[i], cov[i], mag[i]
        )

    # solve cabbages
    mean[mean.isnan()] = init[0][mean.isnan()]
    cov[cov.isnan()] = init[1][cov.isnan()]
    mag[mag.isnan()] = init[2][mag.isnan()]

    return mean, cov, mag, infodict


def _fit_gaussians_no_batch(
    roi: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, mag: torch.Tensor
):
    """Help ``fit_gaussians``."""
    optimizer = torch.optim.LBFGS(
        [mean, cov, mag],
        max_iter=100,
        history_size=6*mag.shape[-1],
        # max_eval=60*mag.shape[-1],  # for case of optimal step only
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        line_search_fn=None,  # not "strong_wolfe" beacause it is not grad efficient,
    )
    shapes = torch.tensor([[roi.shape[0], roi.shape[1]]], dtype=torch.int32, device=roi.device)

    def closure():
        objective, mean_grad, cov_grad, mag_grad = cost_and_grad(
            roi.unsqueeze(0), shapes, mean.unsqueeze(0), cov.unsqueeze(0), mag.unsqueeze(0)
        )
        mean.grad = mean_grad.squeeze(0)
        cov.grad = cov_grad.squeeze(0)
        mag.grad = mag_grad.squeeze(0)
        return objective.mean()

    optimizer.step(closure)

    # print(mean_ := mean.ravel().clone())
    # orig_loss = optimizer.step(closure)
    # print("-----------------------", roi.shape)
    # print("orig loss", optimizer.step(closure))
    # print("n iter:", optimizer.state_dict()["state"][0]["n_iter"])
    # print(optimizer.state_dict()["state"][0].get("prev_loss"))  # can be nan
    # print(":", (mean.ravel()-mean_).tolist())

    return mean, cov, mag
