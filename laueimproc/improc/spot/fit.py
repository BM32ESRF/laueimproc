#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import typing

import numpy as np
import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gmm import cost_and_grad
from laueimproc.gmm.linalg import cov2d_to_eigtheta
from laueimproc.opti.rois import rawshapes2rois


def fit_gaussians_em(
    rois: torch.Tensor,
    photon_density: typing.Union[float, torch.Tensor] = 1.0,
    *,
    eigtheta: bool = False,
    tol: bool = False,
    **extra_info,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    rois : torch.Tensor
        The tensor of the regions of interest for each spots. Shape (n, h, w).
    photon_density : float or Tensor
        Convertion factor to transform the intensity of a pixel
        into the number of photons that hit it.
        Note that the range of intensity values is between 0 and 1.
    eigtheta : boolean, default=False
        If set to True, call ``laueimproc.gmm.linalg.cov2d_to_eigtheta`` and append the result
        in the field `eigtheta` of `infodict`. It is like a PCA on each fit.
    tol : boolean, default=False
        If set to True, the accuracy measure is happend into `infodict`. Shape (n,).
        It correspond to the standard deviation of the estimation of the mean.
        Only available for 1 cluster.
        Let \(\widehat{\mathbf{\mu}}\) be the estimator of the mean.

        * \(\begin{cases}
            std(\widehat{\mathbf{\mu}}) = \sqrt{var(\widehat{\mathbf{\mu}})} \\
            var(\widehat{\mathbf{\mu}}) = var\left( \frac{
                    \sum\limits_{i=1}^N \alpha_i \mathbf{x}_i
                }{
                    \sum\limits_{i=1}^N \alpha_i
                } \right) = \frac{
                    \sum\limits_{i=1}^N var(\alpha_i \mathbf{x}_i)
                }{
                    \left( \sum\limits_{i=1}^N \alpha_i \right)^2
                } \\
            var(\alpha_i \mathbf{x}_i) = var(
                \mathbf{x}_{i1} + \mathbf{x}_{i2} + \dots + \mathbf{x}_{i\alpha_i}) = \alpha_i var(
                \mathbf{x}_i) && \text{because } \mathbf{x}_{ij} \text{ are i.i.d photons} \\
            var(\mathbf{x}_i) = max(eigen( \mathbf{\Sigma} )) \\
        \end{cases}\)
        * \(
            var(\widehat{\mathbf{\mu}})
            = \frac{ max(eigen(\mathbf{\Sigma})) }{ \sum\limits_{i=1}^N \alpha_i }
        \)
        * \(
            std(\widehat{\mathbf{\mu}})\
            = \sqrt{ \frac{ max(eigen(\mathbf{\Sigma})) }{ \sum\limits_{i=1}^N \alpha_i } }
        \)
    **extra_infos : dict
        See ``laueimproc.improc.gmm.em`` for the available metrics.

    Returns
    -------
    mean : torch.Tensor
        The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2). In the relative rois base.
    cov : torch.Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
    eta : torch.Tensor
        The relative mass \(\eta\). Shape (n, \(K\)).
    infodict : dict[str]
        A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
    """
    # verification
    assert isinstance(rois, torch.Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert isinstance(photon_density, (float, torch.Tensor)), photon_density.__class__.__name__
    if isinstance(photon_density, torch.Tensor):
        assert photon_density.shape == (rois.shape[0],), photon_density.shape
    assert isinstance(eigtheta, bool), eigtheta.__class__.__name__
    assert isinstance(tol, bool), tol.__class__.__name__

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(1), points_j.unsqueeze(1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    weights = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))
    weights = weights * photon_density  # copy (no inplace) for keeping rois unchanged

    # fit gaussians
    mean, cov, eta, infodict = em(obs, weights, **extra_info)

    # change base PCA
    if eigtheta:
        infodict["eigtheta"] = cov2d_to_eigtheta(cov)

    # estimation of mean tolerancy
    if tol:
        assert eta.shape[-1] == 1
        std_of_mean = infodict.get(
            "eigtheta", cov2d_to_eigtheta(cov.squeeze(-3), theta=False)
        )[:, 0]
        std_of_mean /= torch.sum(weights, dim=-1)
        std_of_mean = torch.sqrt(std_of_mean, out=std_of_mean)
        infodict["tol"] = std_of_mean

    # cast
    return mean.squeeze(3), cov, eta, infodict


def fit_gaussians(
    data: bytearray, shapes: np.ndarray[np.int32], **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
    shapes : np.ndarray[np.int32]
        Contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.
    **kwargs : dict
        Transmitted to ``laueimproc.improc.gmm.em``, used for initialisation.

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
    mean, cov, mag, infodict = em(
        obs, torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2])), **kwargs
    )
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
