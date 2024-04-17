#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import time
import typing

import numpy as np
import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gmm import cost_and_grad
from laueimproc.gmm.linalg import cov2d_to_eigtheta
from laueimproc.opti.fit import find_optimal_step
from laueimproc.opti.rois import rawshapes2rois


def fit_gaussian_em(
    rois: torch.Tensor,
    photon_density: typing.Union[float, torch.Tensor] = 1.0,
    *,
    tol: bool = False,
    **extra_info,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by one gaussian.

    Based ``laueimproc.improc.spot.fit.fit_gaussians_em`` but squeeze the \(K = 1\) dimension.

    Parameters
    ----------
    rois : torch.Tensor
        Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians_em``.
    photon_density : float or Tensor
        Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians_em``.
    tol : boolean, default=False
        If set to True, the accuracy measure is happend into `infodict`. Shape (n,).
        It correspond to the standard deviation of the estimation of the mean.
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
        Transmitted to ``laueimproc.improc.spot.fit.fit_gaussians_em``.

    Returns
    -------
    mean : torch.Tensor
        The vectors \(\mathbf{\mu}\). Shape (nb_spots, 2). In the relative roi base.
    cov : torch.Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (nb_spots, 2, 2).
    infodict : dict[str]
        A dictionary of optional outputs (see ``laueimproc.improc.spot.fit.fit_gaussians_em``).
    """
    assert isinstance(tol, bool), tol.__class__.__name__

    mean, cov, _, infodict = fit_gaussians_em(rois, photon_density, **extra_info, nbr_clusters=1)

    # squeeze k
    mean, cov = mean.squeeze(1), cov.squeeze(1)
    if "eigtheta" in infodict:
        infodict["eigtheta"] = infodict["eigtheta"].squeeze(1)

    # estimation of mean tolerancy
    if tol:
        weights = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))
        weights = weights * photon_density  # copy (no inplace) for keeping rois unchanged
        std_of_mean = infodict.get("eigtheta", cov2d_to_eigtheta(cov, theta=False))[:, 0]
        std_of_mean /= torch.sum(weights, dim=-1)
        std_of_mean = torch.sqrt(std_of_mean, out=std_of_mean)
        infodict["tol"] = std_of_mean

    # cast
    return mean, cov, infodict


def fit_gaussians_em(
    rois: torch.Tensor,
    photon_density: typing.Union[float, torch.Tensor] = 1.0,
    *,
    eigtheta: bool = False,
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

    # cast
    return mean.squeeze(3), cov, eta, infodict


def fit_gaussians_(
    data: bytearray, shapes: np.ndarray[np.int32],
    loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    eigtheta: bool = False,
    **kwargs,
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
    loss : callable
        Reduce the rois and the predected rois into a single vector of scalar.
        [Tensor(n, h, w), Tensor(n, h, w)] -> Tensor(n,)
    eigtheta : boolean, default=False
        If set to True, call ``laueimproc.gmm.linalg.cov2d_to_eigtheta`` and append the result
        in the field `eigtheta` of `infodict`. It is like a PCA on each fit.
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
    # t_start = time.time()
    # rois = rawshapes2rois(data, shapes.numpy(force=True))
    # mean = torch.empty((len(shapes), kwargs["nbr_clusters"], 2, 1), dtype=torch.float32)
    # cov = torch.empty((len(shapes), kwargs["nbr_clusters"], 2, 2), dtype=torch.float32)
    # mag = torch.empty((len(shapes), kwargs["nbr_clusters"]), dtype=torch.float32)
    # for i, (height, width) in enumerate(shapes.tolist()):
    #     roi = rois[i, :height, :width]
    #     mean[i], cov[i], mag[i] = _fit_gaussians_no_batch(roi, loss, **kwargs)
    # print("t final:", time.time()-t_start)
    # return mean, cov, mag, {}

    # verification
    assert isinstance(eigtheta, bool), eigtheta.__class__.__name__
    assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__

    rois = rawshapes2rois(data, shapes.numpy(force=True))
    print("rois shape", rois.shape)

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    weights = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))  # (n_spots, n_obs)

    # initialization
    mean, cov, mag, infodict = em(obs, weights, **kwargs)
    mag *= torch.amax(weights.unsqueeze(-1), dim=-2)

    t_start = time.time()

    # declaration
    def fold(mean_: torch.Tensor, cov_: torch.Tensor, mag_: torch.Tensor) -> torch.Tensor:
        """Concatenate the 3 values as a simple vector as a copy."""
        return torch.cat((
            mean_.reshape(*mean_.shape[:-3], -1),  # (..., n_clu, 2, 1) -> (..., 2*n_clu)
            cov_.reshape(*cov_.shape[:-3], -1),  # (..., n_clu, 2, 2) -> (..., 4*n_clu)
            mag_,  # (..., n_clu) -> (..., n_clu)
        ), axis=-1)  # (..., 7*n_clu)

    # def unfold(meancovmag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Split the 3 values into 3 componants as a view."""
    #     *batch, n_clu = meancovmag.shape
    #     n_clu //= 7
    #     return (
    #         meancovmag[..., :2*n_clu].reshape(*batch, n_clu, 2, 1),
    #         meancovmag[..., 2*n_clu:6*n_clu].reshape(*batch, n_clu, 2, 2),
    #         meancovmag[..., 6*n_clu:],
    #     )

    prev_pos, curr_pos = fold(mean, cov, mag), None
    prev_cost = curr_cost = None
    grad = None
    ongoing = torch.full((len(shapes),), True, dtype=bool)

    # warming
    prev_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, shapes, loss, mean, cov, mag)
    print(f"cost 1: {prev_cost.mean().item()}")
    mean -= 1e-2 * mean_grad
    cov -= 1e-2 * cov_grad
    mag -= 1e-2 * mag_grad

    curr_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, shapes, loss, mean, cov, mag)
    print(f"cost 2: {curr_cost.mean().item()}")
    curr_pos = fold(mean, cov, mag)
    grad = fold(mean_grad, cov_grad, mag_grad)

    # overfiting
    for i in range(10):
        delta = find_optimal_step(prev_pos-curr_pos, prev_cost-curr_cost, grad)
        # delta = find_optimal_step((prev_pos-curr_pos)[ongoing], (prev_cost-curr_cost)[ongoing], grad)

        # grad_norm = torch.sqrt(torch.sum(grad*grad, dim=-1))
        # print(abs(delta.unsqueeze(-1)*grad).max())
        # delta = delta.clamp(max=1)
        # print("delta min/max:", delta.min(), delta.max())
        # print(torch.sqrt(torch.mean((delta.reshape(-1, 1, 1, 1) * mean_grad)**2)))
        # print("max mean grad:", torch.abs(mean_grad).amax(dim=(-1, -2, -3)).mean().item())
        # print("pos step", torch.abs(
        #     torch.clamp(delta.reshape(-1, 1, 1, 1) * mean_grad, min=-0.1, max=0.1)).amax(dim=(-1, -2, -3)).max()
        # )

        mean -= torch.clamp(delta.reshape(-1, 1, 1, 1) * mean_grad, min=-0.1, max=0.1)
        cov -= torch.clamp(delta.reshape(-1, 1, 1, 1) * cov_grad, min=-0.5, max=0.5)
        mag -= torch.clamp(delta.reshape(-1, 1) * mag_grad, min=-0.05, max=0.05)

        # fact = 1 - i*(1-0.1)/99
        # mean -= torch.clamp(delta.reshape(-1, 1, 1, 1) * mean_grad, min=-0.5*fact, max=0.5*fact)
        # cov -= torch.clamp(delta.reshape(-1, 1, 1, 1) * cov_grad, min=-2*fact, max=2*fact)
        # mag -= torch.clamp(delta.reshape(-1, 1) * mag_grad, min=-0.2*fact, max=0.2*fact)

        # correction
        cov[..., 0, 0] = torch.clamp(cov[..., 0, 0], min=0.1)
        cov[..., 1, 1] = torch.clamp(cov[..., 1, 1], min=0.1)
        sig_prod = cov[..., 0, 0] * cov[..., 1, 1]
        cov[..., 0, 1] = torch.where(
            cov[..., 0, 1]**2 < sig_prod,
            cov[..., 0, 1],
            0.95*torch.sign(cov[..., 0, 1])*torch.sqrt(sig_prod),
        )
        cov[..., 1, 0] = cov[..., 0, 1]

        # # selection
        # if i >= 2:
        #     # ongoing_ = (delta.reshape(-1, 1, 1, 1) * mean_grad).abs().amax(dim=(-1, -2, -3)) >= 1e-4
        #     ongoing = (prev_cost - curr_cost) / prev_cost >= 1e-3
        #     # ongoing = torch.logical_or(ongoing, ongoing_, out=ongoing)
        #     # print(ongoing.to(int).sum())
        #     print(ongoing.to(int).sum())
        #     # ongoing = ongoing_

        # shift history
        prev_pos, prev_cost = curr_pos, curr_cost
        curr_pos = fold(mean, cov, mag)
        grad = fold(mean_grad, cov_grad, mag_grad)[ongoing]

        # print(ongoing.to(int))
        # print(curr_cost[ongoing].shape, curr_cost.shape)
        curr_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, shapes, loss, mean, cov, mag)
        # print(
        #     cost_and_grad(rois[ongoing], shapes[ongoing], loss, mean[ongoing], cov[ongoing], mag[ongoing])[0]
        # )

        print(f"cost {i+2:>2}: {curr_cost.mean().item():e}")

    print("t final:", time.time()-t_start)

    return mean, cov, mag, infodict


def fit_gaussians(
    data: bytearray, shapes: np.ndarray[np.int32],
    loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    eigtheta: bool = False,
    **kwargs,
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
    loss : callable
        Reduce the rois and the predected rois into a single vector of scalar.
        [Tensor(n, h, w), Tensor(n, h, w)] -> Tensor(n,)
    eigtheta : boolean, default=False
        If set to True, call ``laueimproc.gmm.linalg.cov2d_to_eigtheta`` and append the result
        in the field `eigtheta` of `infodict`. It is like a PCA on each fit.
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
    assert isinstance(eigtheta, bool), eigtheta.__class__.__name__
    assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__

    rois = rawshapes2rois(data, shapes.numpy(force=True))

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    weights = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))  # (n_spots, n_obs)

    # initial guess
    mean, cov, mag, infodict = em(obs, weights, **kwargs)
    mag *= torch.amax(weights.unsqueeze(-1), dim=-2)
    mean_, cov_, mag_ = mean.clone(), cov.clone(), mag.clone()

    # overfit
    # t_start = time.time()
    for i, (height, width) in enumerate(shapes.tolist()):
        # print(i)
        mean[i], cov[i], mag[i] = _fit_gaussians_no_batch(
            rois[i, :height, :width], loss, mean[i], cov[i], mag[i]
        )

    cond = mean.isnan()
    mean[cond] = mean_[cond]
    cond = cov.isnan()
    cov[cond] = cov_[cond]
    cond = mag.isnan()
    mag[cond] = mag_[cond]

    # print("t final:", time.time()-t_start)

    return mean, cov, mag, infodict


def _fit_gaussians_no_batch(
    roi: torch.Tensor, loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mean: torch.Tensor, cov: torch.Tensor, mag: torch.Tensor
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
            roi.unsqueeze(0), shapes, loss, mean.unsqueeze(0), cov.unsqueeze(0), mag.unsqueeze(0)
        )
        mean.grad = mean_grad.squeeze(0)
        cov.grad = cov_grad.squeeze(0)
        mag.grad = mag_grad.squeeze(0)
        return objective.mean()

    # print(mean_ := mean.ravel().clone())
    orig_loss = optimizer.step(closure)
    # print("-----------------------", roi.shape)
    # print("orig loss", optimizer.step(closure))
    # print("n iter:", optimizer.state_dict()["state"][0]["n_iter"])
    # print(optimizer.state_dict()["state"][0].get("prev_loss"))  # can be nan
    # print(":", (mean.ravel()-mean_).tolist())

    return mean, cov, mag
