#!/usr/bin/env python3

"""Helper for compute a mixture of multivariate gaussians."""

import torch

from .check import check_gmm
from .gauss import gauss2d, gauss2d_and_jac


def gmm2d(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, eta: torch.Tensor,
    *, _check: bool = True,
) -> torch.Tensor:
    r"""Compute the weighted sum of several 2d gaussians.

    Parameters
    ----------
    obs : torch.Tensor
        The observations of shape (..., \(N\), 2).
        \(\mathbf{x}_i = \begin{pmatrix} x_1 \\ x_2 \\ \end{pmatrix}_j\)
    mean : torch.Tensor
        The 2x1 column mean vector of shape (..., \(K\), 2, 1).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (..., \(K\), 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    eta : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    prob_cum
        The weighted sum of the proba of each gaussian for each observation.
        The shape of \(\Gamma(\mathbf{x}_i)\) if (..., \(N\)).
    """
    if _check:
        check_gmm((mean, cov, eta))

    post = gauss2d(obs, mean, cov, _check=_check)  # (..., n_clu, n_obs)
    prob = torch.sum(post * eta.unsqueeze(-1), dim=-2)  # (..., n_obs)
    return prob


def gmm2d_and_jac(
    obs: torch.Tensor, mean: torch.Tensor, half_cov: torch.Tensor, eta: torch.Tensor,
    *, _check: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the grad of a 2d mixture gaussian model.

    Parameters
    ----------
    obs : torch.Tensor
        The observations of shape (..., \(N\), 2).
        \(\mathbf{x}_i = \begin{pmatrix} x_1 \\ x_2 \\ \end{pmatrix}_j\)
    mean : torch.Tensor
        The 2x1 column mean vector of shape (..., \(K\), 2, 1).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    half_cov : torch.Tensor
        The 2x2 half covariance matrix of shape (..., \(K\), 2, 2).
        \(
            \frac{\mathbf{\Sigma}}{2} + \left(\frac{\mathbf{\Sigma}}{2}\right)^\intercal
            = \mathbf{\Sigma}
            = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}
        \)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    eta : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    prob : torch.Tensor
        Returned value from ``gmm2d`` of shape (..., \(N\))..
    mean_grad : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (..., \(N\), \(K\), 2, 1).
    half_cov_grad : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (..., \(N\), \(K\), 2, 2).
    eta_grad : torch.Tensor
        The gradient of the scalar mass of each gaussian of shape (..., \(N\), \(K\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import gmm2d_and_jac
    >>> obs = torch.randn((1000, 10, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>> half_cov = cov / 2
    >>> eta = torch.rand((1000, 3))  # (..., n_clu)
    >>> eta /= eta.sum(dim=-1, keepdim=True)
    >>>
    >>> prob, mean_jac, half_cov_jac, eta_jac = gmm2d_and_jac(obs, mean, half_cov, eta)
    >>> prob.shape
    torch.Size([1000, 10])
    >>> mean_jac.shape
    torch.Size([1000, 10, 3, 2, 1])
    >>> half_cov_jac.shape
    torch.Size([1000, 10, 3, 2, 2])
    >>> eta_jac.shape
    torch.Size([1000, 10, 3])
    >>>
    """
    if _check:
        check_gmm((mean, half_cov, eta))

    # automatic torch diferenciation (17% slowler than analytic expression)
    # *batch, n_obs, _ = obs.shape
    # *_, n_clu, _, _ = mean.shape
    # obs = obs.reshape(-1, n_obs, 2)  # (b, n_obs, 2)
    # mean = mean.reshape(-1, n_clu, 2, 1)  # (b, n_clu, 2, 1)
    # half_cov = half_cov.reshape(-1, n_clu, 2, 2)  # (b, n_clu, 2, 1)
    # eta = eta.reshape(-1, n_clu)  # (b, n_clu)

    # prob = gmm2d(obs, mean, half_cov + half_cov.mT, eta, _check=_check)
    # prob = prob.reshape(*batch, n_obs)
    # mean_jac, half_cov_jac, eta_jac = (
    #     torch.func.vmap(torch.func.jacfwd(
    #         lambda o, m, hv, e: gmm2d(o, m, hv+hv.mT, e, _check=False),
    #         argnums=(1, 2, 3),
    #     ))(obs, mean, half_cov, eta)
    # )
    # mean_jac = mean_jac.reshape(*batch, n_obs, n_clu, 2, 1)
    # half_cov_jac = half_cov_jac.reshape(*batch, n_obs, n_clu, 2, 2)
    # eta_jac = eta_jac.reshape(*batch, n_obs, n_clu)
    # return prob, mean_jac, half_cov_jac, eta_jac


    *batch, n_clu, _, _ = mean.shape

    prob, mean_jac, half_cov_jac = gauss2d_and_jac(obs, mean, half_cov, _check=_check)
    eta_jac = torch.transpose(prob, -1, -2)
    prob = torch.sum(prob * eta.unsqueeze(-1), dim=-2)  # (..., n_clu, n_obs) -> (..., n_obs)
    mean_jac = torch.sum(mean_jac * eta.reshape(*batch, n_clu, 1, 1, 1, 1), dim=-5)
    half_cov_jac = torch.sum(half_cov_jac * eta.reshape(*batch, n_clu, 1, 1, 1, 1), dim=-5)

    return prob, mean_jac, half_cov_jac, eta_jac
