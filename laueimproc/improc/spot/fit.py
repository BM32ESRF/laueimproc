#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import time
import typing

import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gmm import gmm2d, cost_and_grad
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

    print("rois shape", rois.shape)
    rois = rois[:, :30, :30]
    # rois = rois.to("cuda")

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

    # declaration
    last_last_lr = last_lr = torch.full((len(rois),), 1e-2, dtype=rois.dtype, device=rois.device)
    last_last_last_cost = last_last_cost = last_cost = None  # the 3 last loss
    last_grad = None

    # warming
    last_last_last_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, "mse", mean, cov, mag)
    print(f"cost 1: {last_last_last_cost.mean().item()}")
    mean -= last_last_lr.reshape(len(rois), 1, 1, 1) * mean_grad
    cov -= last_last_lr.reshape(len(rois), 1, 1, 1) * cov_grad
    mag -= last_last_lr.reshape(len(rois), 1) * mag_grad

    last_last_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, "mse", mean, cov, mag)
    print(f"cost 2: {last_last_cost.mean().item()}")
    mean -= last_lr.reshape(len(rois), 1, 1, 1) * mean_grad
    cov -= last_lr.reshape(len(rois), 1, 1, 1) * cov_grad
    mag -= last_lr.reshape(len(rois), 1) * mag_grad
    last_grad = (mean_grad, cov_grad, mag_grad)

    last_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, "mse", mean, cov, mag)
    print(f"cost 3: {last_cost.mean().item()}")

    # overfiting
    for _ in range(50):
        # compute optimal step
        c23 = last_last_cost - last_last_last_cost
        c13 = last_cost - last_last_last_cost
        l21 = last_last_lr + last_lr
        l2 = last_last_lr
        best_lr = .5 * (c23*l21**2 - c13*l2**2) / (c23*l21 - c13*l2) - l21
        best_lr = torch.clamp(best_lr, min=1e-2, max=10.0)  # reduce the risk to go in cabbage
        best_lr = torch.nan_to_num(best_lr, nan=1e-2, out=best_lr)

        # fix two big last step
        in_cabbage = last_cost > last_last_cost
        mean_grad[in_cabbage] = -last_grad[0][in_cabbage]
        cov_grad[in_cabbage] = -last_grad[1][in_cabbage]
        mag_grad[in_cabbage] = -last_grad[2][in_cabbage]
        best_lr[in_cabbage] = last_lr[in_cabbage] - 1e-2

        # update mean, cov and mag
        mean -= best_lr.reshape(len(rois), 1, 1, 1) * mean_grad
        cov -= best_lr.reshape(len(rois), 1, 1, 1) * cov_grad
        mag -= best_lr.reshape(len(rois), 1) * mag_grad

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

        # shift history
        last_last_lr, last_lr = last_lr, torch.where(in_cabbage, 1e-2, best_lr)
        last_last_last_cost, last_last_cost = last_last_cost, last_cost
        last_grad = (  # wrong for in_cabbage but it doesn't matter because not used in next step
            mean_grad, cov_grad, mag_grad
        )

        # compute new grad
        last_cost, mean_grad, cov_grad, mag_grad = cost_and_grad(rois, "mse", mean, cov, mag)
        print(f"cost n: {last_cost.mean().item()}")



    raise NotImplementedError


    t_start = time.time()

    for _ in range(25):
        mean_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * mean_jac, dim=-4)
        half_cov_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * half_cov_jac, dim=-4)
        mag_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1) * mag_jac, dim=-2)
        mean_grad = torch.clamp(mean_grad, -1.0, 1.0)
        half_cov_grad = torch.clamp(half_cov_grad, -1.0, 1.0)
        mag_grad = torch.clamp(mag_grad, -1.0, 1.0)

        # compute optimal step
        c23 = past_cost[-2] - past_cost[-3]
        c13 = past_cost[-1] - past_cost[-3]
        l21 = past_lr[-2] + past_lr[-1]
        l2 = past_lr[-2]
        pos = .5 * (c23*l21**2 - c13*l2**2) / (c23*l21 - c13*l2)
        best_step = pos - l21
        best_step = torch.nan_to_num(best_step, nan=1e-2, posinf=10.0, neginf=1e-2)
        best_step = torch.clamp(best_step, 1e-2, 10.0)
        step = 0.6 * best_step
        # print(best_step)
        # print("grad max", abs(mean_grad).max(), abs(half_cov_grad).max(), abs(mag_grad).max())

        mean -= step.reshape(len(rois), 1, 1, 1) * mean_grad
        half_cov -= step.reshape(len(rois), 1, 1, 1) * half_cov_grad
        mag -= step.reshape(len(rois), 1) * mag_grad
        past_lr.append(step)

        # correction
        half_cov[..., 0, 0] = torch.clamp(half_cov[..., 0, 0], min=0.3)
        half_cov[..., 1, 1] = torch.clamp(half_cov[..., 1, 1], min=0.3)
        sig_prod = half_cov[..., 0, 0] * half_cov[..., 1, 1]
        half_cov[..., 0, 1] = torch.where(
            half_cov[..., 0, 1]**2 < sig_prod,
            half_cov[..., 0, 1],
            0.95*torch.sign(half_cov[..., 0, 1])*torch.sqrt(sig_prod),
        )
        half_cov[..., 1, 0] = half_cov[..., 0, 1]

        # print("min sigma", min(half_cov[..., 0, 0].min(), half_cov[..., 1, 1].min()))
        # print("param max", abs(mean).max(), abs(half_cov).max(), abs(mag).max())

        rois_pred, mean_jac, half_cov_jac, mag_jac = gmm2d_and_jac(obs, mean, half_cov, mag, _check=False)
        rois_pred.grad = 2 * (rois_pred - rois.reshape(*rois_pred.shape))
        past_cost.append(((rois.reshape(*rois_pred.shape) - rois_pred)**2).sum(dim=1))
        print("cost", past_cost[-1].sum().item())


        # # drop converged clusters
        # converged = (past_cost[-2] - past_cost[-1]) / past_cost[-1] < 0.001
        # if torch.any(converged):
        #     if torch.all(converged):
        #         break
        #     print(len(converged))
        #     ongoing = ~converged
        #     obs = obs[ongoing]
        #     mean = mean[ongoing]
        #     half_cov = half_cov[ongoing]
        #     mag = mag[ongoing]
        #     rois_pred_grad = rois_pred.grad[ongoing]
        #     rois_pred = rois_pred[ongoing]
        #     rois_pred.grad = rois_pred_grad
        #     rois = rois[ongoing]
        #     past_cost[-1] = past_cost[-1][ongoing]
        #     past_cost[-2] = past_cost[-2][ongoing]
        #     past_cost[-3] = past_cost[-3][ongoing]
        #     past_lr[-1] = past_lr[-1][ongoing]
        #     past_lr[-2] = past_lr[-2][ongoing]
        #     mean_jac, half_cov_jac, mag_jac = mean_jac[ongoing], half_cov_jac[ongoing], mag_jac[ongoing]



    print("t final:", time.time()-t_start)

    return mean, half_cov+half_cov.mT, mag, infodict
