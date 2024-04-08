#!/usr/bin/env python3

"""Helper for compute a mixture of multivariate gaussians."""

import torch

from .check import check_gmm
from .gauss import gauss2d, gauss2d_and_grad


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

    post = gauss2d(obs, mean, cov)  # (..., n_clu, n_obs)
    prob = torch.sum(post * eta.unsqueeze(-1), dim=-2)  # (..., n_obs)
    return prob


def gmm2d_and_grad(
    obs: torch.Tensor, mean: torch.Tensor, half_cov: torch.Tensor, eta: torch.Tensor,
    *, _check: bool = True,
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
        Returned value from ``gmm2d``.
    mean_grad : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (..., \(K\), 2, 1).
    half_cov_grad : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (..., \(K\), 2, 2).
    eta_grad : torch.Tensor
        The gradient of the scalar mass of each gaussian of shape (..., \(K\)).
    """
    if _check:
        check_gmm((mean, half_cov, eta))

    prob, grad_mean, grad_half_cov = gauss2d_and_grad(obs, mean, half_cov)
    grad_mean = grad_mean * eta.unsqueeze(-1).unsqueeze(-1)
    grad_half_cov = grad_mean * eta.unsqueeze(-1).unsqueeze(-1)
    grad_eta = torch.sum(prob, axis=-1)
    return prob, grad_mean, grad_half_cov, grad_eta
