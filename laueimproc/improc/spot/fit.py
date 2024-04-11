#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import time
import typing

import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gmm import gmm2d, gmm2d_and_jac
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


def fit_gaussians_scipy(
    rois: torch.Tensor,
    loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Fit gaussian with scipy.optimize.minimize."""
    from scipy.optimize import leastsq

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
    mean, half_cov, magnitude, infodict = em(obs, weights, **kwargs)
    half_cov *= 0.5  # for cov = half_cov + half_cov.mT, gradient coefficients symetric
    magnitude *= torch.amax(weights.unsqueeze(-1), dim=-2)

    def func(x, rois, obs, n_clu):
        n = len(rois)
        x = torch.from_numpy(x)
        mean = x[:n*2*n_clu].reshape(n, n_clu, 2, 1)
        half_cov = x[n*2*n_clu:n*2*n_clu + n*4*n_clu].reshape(n, n_clu, 2, 2)
        mag = x[n*2*n_clu + n*4*n_clu:n*2*n_clu + n*4*n_clu + n*n_clu].reshape(n, n_clu)
        rois_pred = gmm2d(obs, mean, half_cov+half_cov.mT, mag, _check=False)
        return (rois.ravel() - rois_pred.ravel()).numpy(force=True)
        # on_cost = loss(rois.reshape(*rois_pred.shape), rois_pred).mean(dim=1)
        # return on_cost.sum().item()

    for i in range(len(rois)):

        x0 = torch.cat([mean[i:i+1].ravel(), half_cov[i:i+1].ravel(), magnitude[i:i+1].ravel()]).numpy(force=True)
        x, cov_x, resinfodict, mesg, ier = leastsq(
            func, x0, args=(rois[i:i+1], obs[i:i+1], kwargs["nbr_clusters"]), xtol=1e-7, full_output=True
        )
        print(f"resultat index {i}, {resinfodict['nfev']} calls, final value {resinfodict['fvec']}")


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
    mean, half_cov, mag, infodict = em(obs, weights, **kwargs)
    half_cov *= 0.5  # for cov = half_cov + half_cov.mT, gradient coefficients symetric
    mag *= torch.amax(weights.unsqueeze(-1), dim=-2)

    # declaration
    past_lr = []  # la liste des vecteurs des learning rates passes, les 2 derniers sont utilises
    past_cost = []  # la liste des loss passes les 3 derniers sont utilises

    # first step

    # compute loss and loss diff
    rois_pred, mean_jac, half_cov_jac, mag_jac = gmm2d_and_jac(obs, mean, half_cov, mag, _check=False)
    # rois_pred = rois_pred.detach()
    # rois_pred.requires_grad = True
    # rois_pred.grad = None
    # cost = loss(rois.reshape(*rois_pred.shape), rois_pred).sum(dim=1)
    # cost.sum().backward()
    rois_pred.grad = 2 * (rois_pred - rois.reshape(*rois_pred.shape))
    past_cost.append(((rois.reshape(*rois_pred.shape) - rois_pred)**2).sum(dim=1))
    print("cost init", past_cost[-1].sum().item())

    # compute grad
    mean_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * mean_jac, dim=-4)
    half_cov_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * half_cov_jac, dim=-4)
    mag_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1) * mag_jac, dim=-2)

    # update values
    mean -= 1e-2 * mean_grad
    half_cov -= 1e-2 * half_cov_grad
    mag -= 1e-2 * mag_grad
    past_lr.append(1e-2)

    # second step
    rois_pred, mean_jac, half_cov_jac, mag_jac = gmm2d_and_jac(obs, mean, half_cov, mag, _check=False)
    rois_pred.grad = 2 * (rois_pred - rois.reshape(*rois_pred.shape))
    past_cost.append(((rois.reshape(*rois_pred.shape) - rois_pred)**2).sum(dim=1))
    print("cost first step", past_cost[-1].sum().item())
    mean_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * mean_jac, dim=-4)
    half_cov_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * half_cov_jac, dim=-4)
    mag_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1) * mag_jac, dim=-2)
    mean -= 1e-2 * mean_grad
    half_cov -= 1e-2 * half_cov_grad
    mag -= 1e-2 * mag_grad
    past_lr.append(1e-2)

    # fird step
    rois_pred, mean_jac, half_cov_jac, mag_jac = gmm2d_and_jac(obs, mean, half_cov, mag, _check=False)
    rois_pred.grad = 2 * (rois_pred - rois.reshape(*rois_pred.shape))
    past_cost.append(((rois.reshape(*rois_pred.shape) - rois_pred)**2).sum(dim=1))
    print("cost second step", past_cost[-1].sum().item())


    t_start = time.time()

    for _ in range(30):
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
        # print(best_step)
        # print("grad max", abs(mean_grad).max(), abs(half_cov_grad).max(), abs(mag_grad).max())

        mean -= 0.6 * best_step.reshape(len(rois), 1, 1, 1) * mean_grad
        half_cov -= 0.6 * best_step.reshape(len(rois), 1, 1, 1) * half_cov_grad
        mag -= 0.6 * best_step.reshape(len(rois), 1) * mag_grad
        past_lr.append(0.6 * best_step)

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

    print("t final:", time.time()-t_start)

    return mean, half_cov+half_cov.mT, mag, infodict


def fit_gaussians_(
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
    mean, half_cov, magnitude, infodict = em(obs, weights, **kwargs)
    half_cov *= 0.5  # for cov = half_cov + half_cov.mT, gradient coefficients symetric
    magnitude *= torch.amax(weights.unsqueeze(-1), dim=-2)

    # print("*********************************************")
    # print("avant", loss_func(mean, half_cov, magnitude, rois, obs).sum().item())

    # raffinement
    ongoing = torch.full((rois.shape[0],), True, dtype=bool)  # mask for clusters converging
    cost = torch.full((rois.shape[0],), torch.inf, dtype=torch.float32)
    rois = rois.detach()
    # lr = torch.full(len(rois), 1e-1, dtype=torch.float32)
    # u, v = torch.zeros(len(rois), dtype=torch.float32), torch.zeros(len(rois), dtype=torch.float32)  # for Adelta optimizer

    for i in range(100):  # not while True for security
        t = time.time()
        on_mean, on_half_cov, on_mag = mean[ongoing], half_cov[ongoing], magnitude[ongoing]
        on_rois, on_obs = rois[ongoing], obs[ongoing]

        # compute loss and loss diff
        rois_pred, mean_jac, half_cov_jac, mag_jac = (
            gmm2d_and_jac(on_obs, on_mean, on_half_cov, on_mag, _check=False)
        )
        rois_pred = rois_pred.detach()
        rois_pred.requires_grad = True
        rois_pred.grad = None
        on_cost = loss(on_rois.reshape(*rois_pred.shape), rois_pred).mean(dim=1)
        on_cost.sum().backward()

        # compute grad
        mean_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * mean_jac, dim=-4)
        half_cov_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1, 1, 1) * half_cov_jac, dim=-4)
        mag_grad = torch.sum(rois_pred.grad.reshape(*rois_pred.shape, 1) * mag_jac, dim=-2)


        print("loss", on_cost.mean().item())
        print("    mean_grad", (mean_grad**2).mean())
        print("    half_cov_grad", (half_cov_grad**2).mean())
        print("    mag_grad", torch.sqrt(mag_grad**2).mean())


        # # drop converged clusters
        # converged = on_cost >= cost[ongoing]
        # cost[ongoing] = on_cost
        # if torch.all(converged):
        #     break

        # update values
        lr = 128 * .933**i  # reduce by 2 every 10 iterations
        mean[ongoing] -= mean_grad * lr  # on_lr.reshape(-1, 1, 1, 1)
        half_cov[ongoing] -= half_cov_grad * lr  # on_lr.reshape(-1, 1, 1, 1)
        magnitude[ongoing] -= mag_grad * lr  # on_lr.reshape(-1, 1)

        # # we want to do equivalent of ongoing[ongoing][converged] = False
        # ongoing_ = ongoing[ongoing]
        # ongoing_[converged] = False
        # ongoing[ongoing.clone()] = ongoing_
    else:
        logging.warning("gaussian convergence cycle prematurely interrupted")

    # print("apres", loss_func(mean, half_cov, magnitude, rois, obs).sum().item())

    cov = half_cov+half_cov.mT

    # change base PCA
    if eigtheta:
        infodict["eigtheta"] = cov2d_to_eigtheta(cov)

    # hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(loss_func, argnums=0), argnums=0))(on_mean, on_half_cov, on_magnitude, rois[ongoing], on_obs)

    return mean, cov, magnitude, infodict
