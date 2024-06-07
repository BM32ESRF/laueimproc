#!/usr/bin/env python3

"""Metric to estimate the quality of the fit of the gmm."""

import math
import typing

import torch

from .check import check_gmm, check_infit
from .gauss import gauss


def aic(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    _llh: typing.Optional[torch.Tensor] = None,
    _check: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the Akaike Information Criterion and the Bayesian Information Criterion.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\)).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    aic : torch.Tensor
        Akaike Information Criterion
        \(aic = 2p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the likelihood.
    """
    if _check:
        check_infit(obs, weights)
        check_gmm(gmm)

    free_parameters: int = (
        (obs.shape[-1] * (obs.shape[-1]+1))**2 // 2  # nbr of parameters in cov matrix
        + obs.shape[-1]  # nbr of parameters in mean vector
        + 1  # eta
    ) * gmm[2].shape[-1]  # nbr of gaussians
    m2llh = log_likelihood(obs, weights, gmm, _check=False) if _llh is None else _llh.clone()
    m2llh *= -2.0
    aic_val = m2llh + 2*float(free_parameters)
    return aic_val


def bic(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    _llh: typing.Optional[torch.Tensor] = None,
    _check: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the Akaike Information Criterion and the Bayesian Information Criterion.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\)).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    bic : torch.Tensor
        Bayesian Information Criterion
        \(bic = \log(N)p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the likelihood.
    """
    if _check:
        check_infit(obs, weights)
        check_gmm(gmm)

    free_parameters: int = (
        (obs.shape[-1] * (obs.shape[-1]+1))**2 // 2  # nbr of parameters in cov matrix
        + obs.shape[-1]  # nbr of parameters in mean vector
        + 1  # eta
    ) * gmm[2].shape[-1]  # nbr of gaussians
    m2llh = log_likelihood(obs, weights, gmm, _check=False) if _llh is None else _llh.clone()
    m2llh *= -2.0

    if weights is None:
        log_n_obs = math.log(obs.shape[-2])
    else:
        log_n_obs = torch.sum(weights, axis=-1, keepdim=False)
        log_n_obs = torch.log(log_n_obs, out=log_n_obs)
    bic_val = m2llh + log_n_obs*float(free_parameters)
    return bic_val


def log_likelihood(
    obs: torch.Tensor,
    weights: typing.Optional[torch.Tensor],
    gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *, _check: bool = True,
) -> torch.Tensor:
    r"""Compute the log likelihood.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\)).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    log_likelihood : torch.Tensor
        The log likelihood \( L_{\alpha,\omega} \) of shape (...,).
        \(
            L_{\alpha,\omega} = \log\left(
                \prod\limits_{i=1}^N \sum\limits_{j=1}^K
                \eta_j \left(
                    \mathcal{N}_{(\mathbf{\mu}_j,\frac{1}{\omega_i}\mathbf{\Sigma}_j)}(\mathbf{x}_i)
                \right)^{\alpha_i}
            \right)
        \)
    """
    if _check:
        check_infit(obs, weights)
        check_gmm(gmm)

    mean, cov, eta = gmm
    prob = gauss(obs, mean, cov, _check=False)  # (..., n_clu, n_obs)
    prob **= weights.unsqueeze(-2)
    prob *= eta.unsqueeze(-1)
    ind_prob = torch.sum(prob, axis=-2, keepdim=False)  # (..., n_obs)
    ind_prob += torch.finfo(torch.float32).eps  # for stability
    ind_prob = torch.log(ind_prob, out=None if ind_prob.requires_grad else ind_prob)
    return torch.sum(ind_prob, axis=-1, keepdim=False)  # (...,)
