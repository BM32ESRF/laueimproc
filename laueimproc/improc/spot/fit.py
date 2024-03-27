#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import logging
import typing

import torch

from laueimproc.gmm.em import em
from laueimproc.gmm.gauss import _gauss2d


def fit_gaussian_em(
    rois: torch.Tensor,
    photon_density: typing.Union[float, torch.Tensor] = 1.0,
    *,
    tol: bool = False,
    **extra_info,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by one gaussian.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    rois : Tensor
        The tensor of the regions of interest for each spots. Shape (n, h, w).
    photon_density : float or Tensor
        Convertion factor to transform the intensity of a pixel
        into the number of photons that hit it.
        Note that the range of intensity values is between 0 and 1.
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
        See ``laueimproc.improc.gmm.em`` for the available metrics.

    Returns
    -------
    mean : Tensor
        The vectors \(\mathbf{\mu}\). Shape (nb_spots, 2). In the relative roi base.
    cov : Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (nb_spots, 2, 2).
    infodict : dict[str]
        A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
    """
    # verification
    assert isinstance(rois, torch.Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert isinstance(photon_density, (float, torch.Tensor)), photon_density.__class__.__name__
    if isinstance(photon_density, torch.Tensor):
        assert photon_density.shape == (rois.shape[0],), photon_density.shape
    assert isinstance(tol, bool), tol.__class__.__name__

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], axis=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    dup_w = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))
    dup_w = dup_w * photon_density  # copy (no inplace) for keeping rois unchanged

    # fit gaussian
    mean, cov, _, infodict = em(obs, dup_w, **extra_info, nbr_clusters=1)

    # estimation of mean tolerancy
    if tol:
        std_of_mean = torch.max(torch.linalg.eigvalsh(cov), axis=-1).values.squeeze(-1)
        std_of_mean /= torch.sum(dup_w, axis=-1)
        std_of_mean = torch.sqrt(std_of_mean, out=std_of_mean)
        infodict["tol"] = std_of_mean

    # cast
    return mean.squeeze(-3).squeeze(-1), cov.squeeze(-3), infodict


def fit_gaussians(
    rois: torch.Tensor, loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""Fit each roi by \(K\) gaussians.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    rois : Tensor
        The tensor of the regions of interest for each spots. Shape (n, h, w).
    loss : callable
        Reduce the rois and the predected rois into a single vector of scalar.
        [Tensor(n, h, w), Tensor(n, h, w)] -> Tensor(n,)
    **kwargs : dict
        Transmitted to ``laueimproc.improc.gmm.em``, used for initialisation.

    Returns
    -------
    mean : Tensor
        The vectors \(\mathbf{\mu}\). Shape (n, \(K\), 2, 1). In the relative rois base.
    cov : Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (n, \(K\), 2, 2).
    mass : Tensor
        The absolute mass \(\theta.\eta\). Shape (n, \(K\)).
    infodict : dict[str]

        * "loss" : Tensor
            The vector of loss. Shape (n,).
        * "pred" : Tensor
            The predicted rois tensor. Shape(n, h, w)
    """
    # verification
    assert isinstance(rois, torch.Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert callable(loss), loss.__class__.__name__

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
    mean, half_cov, mass, infodict = em(obs, dup_w, **kwargs)
    half_cov *= 0.5  # for cov = half_cov + half_cov.mT, gradient coefficients symetric
    mass *= torch.sum(dup_w.unsqueeze(-1), axis=-2)

    # @torch.compile(fullgraph=True)
    def loss_func(mean_, half_cov_, mass_, rois_, obs_):
        pred_rois_ = torch.sum(
            (_gauss2d(obs_, mean_, half_cov_+half_cov_.mT) * mass_.unsqueeze(-1)), -2
        ).reshape(rois_.shape)
        cost_ = loss(pred_rois_, rois_)
        return cost_

    print("avant", loss_func(mean, half_cov, mass, rois, obs).sum())

    # raffinement
    ongoing = torch.full((rois.shape[0],), True, dtype=bool)  # mask for clusters converging
    cost = torch.full((rois.shape[0],), torch.inf, dtype=torch.float32)
    for _ in range(1000):  # not while True for security
        on_obs = obs[ongoing]
        on_mean, on_half_cov, on_mass = mean[ongoing], half_cov[ongoing], mass[ongoing]

        # compute grad
        on_mean.requires_grad = on_half_cov.requires_grad = on_mass.requires_grad = True
        on_mean.grad = on_half_cov.grad = on_mass.grad = None
        loss_func(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs).sum().backward()

        # optimal step
        on_lr = torch.zeros((rois.shape[0],), dtype=torch.float32, requires_grad=True)
        on_mean.requires_grad = on_half_cov.requires_grad = on_mass.requires_grad = False
        on_cost = loss_func(
            on_mean - on_lr.reshape(-1, 1, 1, 1)*on_mean.grad,
            on_half_cov - on_lr.reshape(-1, 1, 1, 1)*on_half_cov.grad,
            on_mass - on_lr.reshape(-1, 1)*on_mass.grad,
            rois[ongoing],
            on_obs,
        )
        first_diff = torch.autograd.grad(on_cost.sum(), on_lr, create_graph=True)[0]
        print(first_diff.shape)
        print(torch.all(first_diff <= 0))
        second_diff = torch.autograd.grad(first_diff.sum(), on_lr)[0]
        print(second_diff.shape)
        print(torch.all(second_diff >= 0))


        # # for _ in range(100):
        # import time
        # ti = time.time()
        # # hessian = torch.func.vmap(torch.func.hessian(loss_, argnums=0))(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs)
        # hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(loss_func, argnums=0), argnums=0))(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs)
        # print("fwd o rev", time.time()-ti)

        # ti = time.time()
        # hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(loss_func, argnums=0), argnums=0))(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs)
        # print("fwd o fwd", time.time()-ti)

        # ti = time.time()
        # hessian = torch.func.vmap(torch.func.jacrev(torch.func.jacrev(loss_func, argnums=0), argnums=0))(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs)
        # print("rev o rev", time.time()-ti)

            # ti = time.time()
            # on_mean.grad = on_half_cov.grad = on_mass.grad = None
            # on_cost = loss_(on_mean, on_half_cov, on_mass, rois[ongoing], on_obs)

            # # on_cost.sum().backward(create_graph=True, retain_graph=True)
            # on_jac_mean = torch.autograd.grad(on_cost, on_mean, create_graph=True)[0]
            # print("jac", on_jac_mean.shape)

            # # hess = torch.zeros_like(hessian)
            # # on_mean.grad = None
            # hessian = torch.autograd.grad(on_jac_mean[0], inputs=on_mean[0], grad_outputs=torch.ones_like(on_mean[0]))
            # print("hess", hessian.shape)
            # on_mean.grad = None
            # on_jac_mean[0].backward(on_mean[0])

            # torch.autograd.backward(on_jac_mean[0], on_mean[0])
            # on_jac_mean[0].backward(on_mean[0])
            # print(on_mean.grad[0].shape)


        # drop converged clusters
        converged = on_cost >= cost[ongoing]
        cost[ongoing] = on_cost
        on_lr[converged] /= 1.2589254117941673  # 10**(1/10)
        on_lr[~converged] *= 1.023292992280754  # 10**(1/100)
        learing_rate[ongoing] = on_lr

        # # update values
        # mean_hess = hessian.reshape(-1, mean.shape[-3]*2, mean.shape[-3]*2)
        # inv_mean_hess = torch.linalg.inv(mean_hess)
        # new_mean = -inv_mean_hess @ on_mean.grad.reshape(-1, mean.shape[-3]*2, 1)
        # new_mean = new_mean.reshape(-1, mean.shape[-3], 2, 1)

        # print("dist:", torch.dist(new_mean, on_mean))
        # mean[ongoing] = new_mean
        # break


        # update values
        mean[ongoing] -= 1e-3*on_mean.grad  # * on_lr.reshape(-1, 1, 1, 1)
        half_cov[ongoing] -= 1e-3*on_half_cov.grad  # * on_lr.reshape(-1, 1, 1, 1)
        mass[ongoing] -= 1e-3*on_mass.grad  # * on_lr.reshape(-1, 1)

        # we want to do equivalent of ongoing[ongoing][converged] = False
        if torch.all(converged):
            break
        ongoing_ = ongoing[ongoing]
        ongoing_[converged] = False
        ongoing[ongoing.clone()] = ongoing_
    else:
        logging.warning("some gmm clusters failed to converge after 1000 iterations")

    print("apres", loss_func(mean, half_cov, mass, rois, obs).sum())

    return mean, half_cov+half_cov.mT, mass, infodict
