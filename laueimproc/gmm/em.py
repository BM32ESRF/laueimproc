#!/usr/bin/env python3

r"""Implement the EM (Esperance Maximisation) algo.

Detailed algorithm for going from step \(s\) to step \(s+1\):

* \(
    p_{i,j} = \frac{
        \eta_j^{(s)}
        \mathcal{N}_{\mathbf{\mu}_j^{(s)}, \mathbf{\Sigma}_j^{(s)}}\left(\mathbf{x}_i\right)
    }{
        \sum\limits_{k=1}^K
        \eta_k^{(s)}
        \mathcal{N}_{\mathbf{\mu}_k^{(s)}, \mathbf{\Sigma}_k^{(s)}}\left(\mathbf{x}_i\right)
    }
\) Posterior probability that observation \(i\) belongs to cluster \(j\).
* \(
    \eta_j^{(s+1)} = \frac{\sum\limits_{i=1}^N \alpha_i p_{i,j}}{\sum\limits_{i=1}^N \alpha_i}
\) The relative weight of each gaussian.
* \(
    \mathbf{\mu}_j^{(s+1)} = \frac{
        \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j} \mathbf{x}_i
    }{
        \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j}
    }
\) The mean of each gaussian.
* \(
    \mathbf{\Sigma}_j^{(s+1)} = \frac{
        \sum\limits_{i=1}^N
        \omega_i \alpha_i p_{i,j}
        \left(\mathbf{x}_i - \mathbf{\mu}_j^{(s+1)}\right)
        \left(\mathbf{x}_i - \mathbf{\mu}_j^{(s+1)}\right)^{\intercal}
    }{
        \sum\limits_{i=1}^N \alpha_i p_{i,j}
    }
\) The cov of each gaussian.

This sted is iterated as long as the log likelihood increases.
"""

import logging
import numbers
import typing

import torch

from .check import check_infit
from .gmm import gmm2d
from .metric import aic_bic, log_likelihood


def _fit_one_cluster(
    obs: torch.Tensor, weights: typing.Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fit one gaussian.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).

    Returns
    -------
    mean : torch.Tensor
        The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(D\), 1).
    cov : torch.Tensor
        The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(D\), \(D\)).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    if weights is None:
        mean = torch.mean(obs, dim=-2, keepdim=True)  # (..., 1, n_var)
    else:
        mean = torch.sum((weights.unsqueeze(-1) * obs), dim=-2, keepdim=True)
        mean /= torch.sum(weights.unsqueeze(-1), dim=-2, keepdim=True)  # (..., 1, n_var)
    cent_obs = (obs - mean).unsqueeze(-1)  # (..., n_obs, n_var, 1)
    mean = mean.squeeze(-2).unsqueeze(-1)  # (..., n_var, 1)
    if weights is None:
        cov = torch.mean(  # (..., n_var, n_var)
            cent_obs @ cent_obs.transpose(-1, -2),
            dim=-3, keepdim=False  # (..., n_obs, n_var, n_var) -> (..., n_var, n_var)
        )
    else:
        cov = torch.sum(  # (..., n_var, n_var)
            weights.unsqueeze(-1).unsqueeze(-1) * cent_obs @ cent_obs.mT,
            dim=-3, keepdim=False  # (..., n_obs, n_var, n_var) -> (..., n_var, n_var)
        )
        cov /= torch.sum(weights, dim=-1, keepdim=False).unsqueeze(-1).unsqueeze(-1)
    return mean, cov


def _fit_n_clusters_one_step(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One step of the EM algo.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\), 1).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    mean : torch.Tensor
        A reference to the input parameter `mean`.
    cov : torch.Tensor
        A reference to the input parameter `cov`.
    eta : torch.Tensor
        A reference to the input parameter `eta`.

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    mean, cov, eta = gmm
    eps = torch.finfo(obs.dtype).eps

    # posterior probability that observation j belongs to cluster i
    post = gmm2d(obs, mean, cov, eta, _check=False)  # (..., n_clu, n_obs)
    post += eps  # prevention again division by 0
    post /= torch.sum(post, dim=-2, keepdim=True)  # (..., n_clu, n_obs)

    # update the relative mass of each gaussians (not normalized because it it used for cov)
    if weights is None:
        eta = torch.sum(post, dim=-1)  # (..., n_clu)
    else:
        eta = torch.sum(weights.unsqueeze(-2) * post, dim=-1, keepdim=False, out=eta)

    # mean
    if weights is None:
        weighted_post = post
        mean = torch.mean(
            obs.unsqueeze(-3).unsqueeze(-1),  # (..., n_clu, n_obs, n_var, 1)
            dim=-3, keepdim=False, out=mean  # (..., n_clus, n_var, 1)
        )
    else:
        weighted_post = weights.unsqueeze(-2) * post  # (..., n_clu, n_obs)
        weighted_post = weighted_post.unsqueeze(-1).unsqueeze(-1)  # (..., n_clu, n_obs, 1, 1)
        mean = torch.sum(
            weighted_post * obs.unsqueeze(-3).unsqueeze(-1),  # (..., n_clu, n_obs, n_var, 1)
            dim=-3, keepdim=False, out=mean  # (..., n_clus, n_var, 1)
        )
        mean /= torch.sum(weighted_post, dim=-3, keepdim=False)  # (..., n_clus, n_var, 1)

    # cov
    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    cov = torch.sum(
        weighted_post * cent_obs @ cent_obs.transpose(-1, -2),  # (..., n_clu, n_obs, n_var, n_var)
        dim=-3, keepdim=False, out=cov  # (..., n_clu, n_var, n_var)
    )
    cov /= eta.unsqueeze(-1).unsqueeze(-1)

    # normalize eta
    if weights is not None:
        eta /= torch.sum(weights.unsqueeze(-2), dim=-1, keepdim=False)  # (..., n_clu)

    return mean, cov, eta


def _fit_n_clusters_serveral_steps(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""N steps of the EM algo.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\), 1).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    gmm : tuple of torch.Tensor
        mean : torch.Tensor
            A reference to the input parameter `mean`.
        cov : torch.Tensor
            A reference to the input parameter `cov`.
        eta : torch.Tensor
            A reference to the input parameter `eta`.
    log_likelihood : torch.Tensor
        The log likelihood of the last result of shape (...,).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    mean, cov, eta = gmm
    llh = log_likelihood(obs, weights, gmm, _check=False)  # (...,)
    ongoing = torch.ones_like(llh, dtype=bool)  # mask for clusters converging

    for _ in range(1000):  # not while True for security
        on_weights = None if weights is None else weights[ongoing]
        new_gmm = _fit_n_clusters_one_step(
            obs[ongoing], on_weights, (mean[ongoing], cov[ongoing], eta[ongoing])
        )
        mean[ongoing], cov[ongoing], eta[ongoing] = new_gmm
        new_llh = log_likelihood(obs[ongoing], on_weights, new_gmm, _check=False)
        converged = (   # the mask of the converged clusters
            new_llh - llh[ongoing] <= 1e-3
        )
        llh[ongoing] = new_llh
        if torch.all(converged):
            break
        # we want to do equivalent of ongoing[ongoing][converged] = False
        ongoing_ = ongoing[ongoing]
        ongoing_[converged] = False
        ongoing[ongoing.clone()] = ongoing_
    else:
        logging.warning("some gmm clusters failed to converge after 1000 iterations")

    return (mean, cov, eta), llh


def _fit_n_clusters_serveral_steps_serveral_tries(
    nbr_tries: int,
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""N steps and serveral tries of the EM algo.

    Keep only the best result for each run.

    Parameters
    ----------
    nbr_tries : int
        The number of times the algo is applied to the same data,
        but with different random initializations.
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\), 1).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    gmm : tuple of torch.Tensor
        mean : torch.Tensor
            A reference to the input parameter `mean`.
        cov : torch.Tensor
            A reference to the input parameter `cov`.
        eta : torch.Tensor
            A reference to the input parameter `eta`.
    log_likelihood : torch.Tensor
        The log likelihood of the last result of shape (...,).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    *batch, n_obs, n_var = obs.shape
    mean, cov, eta = gmm
    nbr_clusters = eta.shape[-1]

    # draw random initial conditions
    mean = (  # (n_tries, ..., n_clu, n_var, 1)
        mean.unsqueeze(-3).unsqueeze(0)
        + (
            torch.randn(
                (nbr_tries, *batch, nbr_clusters, n_var),
                dtype=obs.dtype, device=obs.device
            ) @ cov
        ).unsqueeze(-1)
    )
    cov = (  # (n_tries, ..., n_clu, n_var, n_var)
        cov
        .unsqueeze(-3).unsqueeze(0)
        .expand(nbr_tries, *batch, nbr_clusters, n_var, n_var)
        .clone()
    )
    eta = ( # (n_tries, ..., n_clu)
        eta.unsqueeze(0).expand(nbr_tries, *batch, nbr_clusters).clone()
    )

    # prepare data for several draw
    obs = obs.unsqueeze(0).expand(nbr_tries, *batch, n_obs, n_var)
    if weights is not None:
        weights = weights.unsqueeze(0).expand(nbr_tries, *batch, n_obs)

    # run em on each cluster and each tries
    (mean, cov, eta), llh = _fit_n_clusters_serveral_steps(obs, weights, (mean, cov, eta))

    # keep the best
    if nbr_clusters == 1:
        mean, cov, eta = mean.unsqueeze(0), cov.unsqueeze(0), eta.unsqueeze(0)
    else:
        llh, best = torch.max(llh, dim=0)
        mean = mean[best, range(mean.shape[1])]
        cov = cov[best, range(cov.shape[1])]
        eta = eta[best, range(eta.shape[1])]
    return (mean, cov, eta), llh


def em(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor] = None,
    *,
    nbr_clusters: numbers.Integral = 1,
    nbr_tries: numbers.Integral = 2,
    **metrics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""A weighted implementation of the EM algorithm.

    Parameters
    ----------
    obs : arraylike
        The observations \(\mathbf{x}\). Shape (..., \(N\), \(D\)).
    weights : arraylike, optional
        The weights \(\alpha\) has the relative number of time the individual has been drawn.
        Shape (..., \(N\)).
    nbr_clusters : int, default=1
        The number \(K\) of gaussians.
    nbr_tries : int, default=2
        The number of times that the algorithm converges
        in order to have the best possible solution.
        It is ignored in the case `nbr_clusters` = 1.
    aic, bic, log_likelihood : boolean, default=False
        If set to True, the metric is happend into `infodict`.

        * aic: Akaike Information Criterion of shape (...,). \(aic = 2p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the log likelihood.
        * bic: Bayesian Information Criterion of shape (...,).
        \(bic = \log(N)p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the log likelihood.
        * log_likelihood of shape (...,): \(
            L_{\alpha,\omega} = \log\left(
                \prod\limits_{i=1}^N \sum\limits_{j=1}^K
                \eta_j \left(
                    \mathcal{N}_{(\mathbf{\mu}_j,\frac{1}{\omega_i}\mathbf{\Sigma}_j)}(\mathbf{x}_i)
                \right)^{\alpha_i}
            \right)
        \)
        * log_likelihood of shape (...,): See ``laueimproc.gmm.metric.log_likelihood``.

    Returns
    -------
    mean : torch.Tensor
        The vectors \(\mathbf{\mu}\). Shape (..., \(K\), \(D\), 1).
    cov : torch.Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (..., \(K\), \(D\), \(D\)).
    eta : torch.Tensor
        The relative mass \(\eta\). Shape (..., \(K\)).
    infodict : dict[str]
        A dictionary of optional outputs.
    """
    m_name = {"aic", "bic", "log_likelihood"}

    # verifications
    check_infit(obs, weights)
    assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
    assert nbr_clusters > 0, nbr_clusters
    nbr_clusters = int(nbr_clusters)
    assert isinstance(nbr_tries, numbers.Integral), nbr_tries.__class__.__name__
    assert nbr_tries > 0, nbr_tries
    nbr_tries = int(nbr_tries)
    assert all(isinstance(metrics.get(k, False), bool) for k in metrics), metrics
    assert set(metrics).issubset(m_name), f"unrecognise parameters {set(metrics)-m_name}"

    # main gmm
    *batch, _, _ = obs.shape
    mean, cov = _fit_one_cluster(obs, weights)  # (..., n_var, 1) and (..., n_var, n_var)
    eta = torch.full((*batch, nbr_clusters), 1.0/nbr_clusters, dtype=obs.dtype, device=obs.device)
    llh = None
    if nbr_clusters != 1:
        (mean, cov, eta), llh = _fit_n_clusters_serveral_steps_serveral_tries(
            nbr_tries, obs, weights, (mean, cov, eta)
        )
    else:  # case one cluster
        mean = mean.unsqueeze(-3)  # (..., 1, n_var, 1)
        cov = cov.unsqueeze(-3)  # (..., 1, n_var, n_var)
        if any(metrics.get(n, False) for n in ("log_likelihood", "aic", "bic")):
            llh = log_likelihood(obs, weights, (mean, cov, eta), _check=False)

    # finalization
    infodict = {}

    if any(metrics.get(n, False) for n in ("aic", "bic")):
        aic_bic_ = aic_bic(obs, weights, (mean, cov, eta), _llh=llh, _check=False)
        if metrics.get("aic", False):
            infodict["aic"] = aic_bic_[0]
        if metrics.get("bic", False):
            infodict["bic"] = aic_bic_[1]
    if metrics.get("log_likelihood", False):
        infodict["log_likelihood"] = llh

    # cast
    return mean, cov, eta, infodict
