#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import time
import typing

import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gauss import gauss2d
from laueimproc.gmm.linalg import cov2d_to_eigtheta


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
        std_of_mean = infodict.get("eigtheta", cov2d_to_eigtheta(cov, theta=False))[:, 0]
        std_of_mean /= torch.sum(dup_w, axis=-1)
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
    obs = torch.cat([points_i.unsqueeze(1), points_j.unsqueeze(1)], axis=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    dup_w = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))
    dup_w = dup_w * photon_density  # copy (no inplace) for keeping rois unchanged

    # fit gaussians
    mean, cov, eta, infodict = em(obs, dup_w, **extra_info)

    # change base PCA
    if eigtheta:
        infodict["eigtheta"] = cov2d_to_eigtheta(cov)

    # cast
    return mean.squeeze(3), cov, eta, infodict


def fit_gaussians(
    rois: torch.Tensor,
    loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    eigtheta: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    rois : torch.Tensor
        The tensor of the regions of interest for each spots. Shape (n, h, w).
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
    assert isinstance(rois, torch.Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert callable(loss), loss.__class__.__name__
    assert isinstance(eigtheta, bool), eigtheta.__class__.__name__

    # declaration
    # @torch.compile(fullgraph=True)
    def loss_func(mean_, half_cov_, mass_, rois_, obs_):
        pred_rois_ = torch.sum(
            (_gauss2d(obs_, mean_, half_cov_+half_cov_.mT) * mass_.unsqueeze(-1)), -2
        ).reshape(rois_.shape)
        # cost_ = loss(pred_rois_, rois_)
        cost_ = torch.mean((pred_rois_ - rois_)**2)
        return cost_

    grad_val_func = torch.func.vmap(torch.func.grad_and_value(loss_func, argnums=(0, 1, 2)))
    lr1_lr2_func = torch.func.vmap(
        torch.func.grad_and_value(
            torch.func.grad(
                lambda mean_, half_cov_, mass_, rois_, obs_, lr_, g_mean_, g_half_cov_, g_mass_: loss_func(
                    mean_ - lr_.reshape(-1, 1, 1, 1)*g_mean_,
                    half_cov_ - lr_.reshape(-1, 1, 1, 1)*g_half_cov_,
                    mass_ - lr_.reshape(-1, 1)*g_mass_,
                    rois_,
                    obs_,
                ),
                argnums=5,
            ),
            argnums=5,
        )
    )

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], axis=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    dup_w = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))  # (n_spots, n_obs)

    # initialization
    mean, half_cov, magnitude, infodict = em(obs, dup_w, **kwargs)
    half_cov *= 0.5  # for cov = half_cov + half_cov.mT, gradient coefficients symetric
    magnitude *= torch.amax(dup_w.unsqueeze(-1), axis=-2)

    # print("*********************************************")
    # print("avant", loss_func(mean, half_cov, magnitude, rois, obs).sum().item())

    # raffinement
    ongoing = torch.full((rois.shape[0],), True, dtype=bool)  # mask for clusters converging
    cost = torch.full((rois.shape[0],), torch.inf, dtype=torch.float32)
    for _ in range(10):  # not while True for security
        t = time.time()
        on_mean, on_half_cov, on_magnitude = mean[ongoing], half_cov[ongoing], magnitude[ongoing]
        on_rois, on_obs = rois[ongoing], obs[ongoing]

        # compute loss and grad
        (grad_on_mean, grad_on_half_cov, grad_on_magnitude), on_cost = grad_val_func(
            on_mean, on_half_cov, on_magnitude, on_rois, on_obs
        )

        # print("cost", cost.sum().item(), "nelem", len(on_rois))

        # optimal step, second order derivated of loss in the grad direction
        on_lr = torch.zeros((on_rois.shape[0],), dtype=torch.float32)
        lr2, lr1 = lr1_lr2_func(
            on_mean, on_half_cov, on_magnitude, on_rois, on_obs, on_lr, grad_on_mean, grad_on_half_cov, grad_on_magnitude
        )
        # print(f"    temps: {time.time()-t:.2f}")
        on_lr = -lr1 / lr2  # assume quadratic model convergence
        # on_lr = torch.clamp(on_lr, 0.0, 1.0)
        # print("step", on_lr.min().item(), on_lr.max().item())

        # drop converged clusters
        converged = on_cost >= cost[ongoing]
        cost[ongoing] = on_cost
        if torch.all(converged):
            break

        # update values
        mean[ongoing] -= grad_on_mean * on_lr.reshape(-1, 1, 1, 1)
        half_cov[ongoing] -= grad_on_half_cov * on_lr.reshape(-1, 1, 1, 1)
        magnitude[ongoing] -= grad_on_magnitude * on_lr.reshape(-1, 1)

        # we want to do equivalent of ongoing[ongoing][converged] = False
        ongoing_ = ongoing[ongoing]
        ongoing_[converged] = False
        ongoing[ongoing.clone()] = ongoing_
    else:
        logging.warning("gaussian convergence cycle prematurely interrupted")

    # print("apres", loss_func(mean, half_cov, magnitude, rois, obs).sum().item())

    cov = half_cov+half_cov.mT

    # change base PCA
    if eigtheta:
        infodict["eigtheta"] = cov2d_to_eigtheta(cov)

    # hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(loss_func, argnums=0), argnums=0))(on_mean, on_half_cov, on_magnitude, rois[ongoing], on_obs)

    return mean, cov, magnitude, infodict
