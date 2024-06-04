#!/usr/bin/env python3

"""Helper for compute multivariate gaussian."""

import torch

from .check import check_ingauss
from .linalg import inv_cov2d


def gauss(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, *, _check: bool = True
) -> torch.Tensor:
    r"""Compute a multivariate gaussian.

    \(
        \mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}(\mathbf{x}_i) =
        \frac
        {
            e^{
                -\frac{1}{2}
                (\mathbf{x}_i-\mathbf{\mu}_j)^\intercal
                \mathbf{\Sigma}_j^{-1}
                (\mathbf{x}_i-\mathbf{\mu}_j)
            }
        }
        {\sqrt{(2\pi)^D |\mathbf{\Sigma}_j}|}
    \)

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    mean : torch.Tensor
        The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\), 1).
    cov : torch.Tensor
        The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).

    Returns
    -------
    prob : torch.Tensor
        The prob density to draw the sample obs of shape (..., \(K\), \(N\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gauss import gauss
    >>> obs = torch.randn((1000, 10, 4))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 4))  # (..., n_clu, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 4, 4)  # (..., n_clu, n_var, n_var)
    >>>
    >>> prob = gauss(obs, mean, cov)
    >>> prob.shape
    torch.Size([1000, 3, 10])
    >>>
    >>> mean.requires_grad = cov.requires_grad = True
    >>> gauss(obs, mean, cov).sum().backward()
    >>>
    """
    if _check:
        check_ingauss(obs, mean, cov)

    norm = torch.linalg.det(cov).unsqueeze(-1)  # (..., n_clu, 1)
    norm = torch.mul(norm, (2.0*torch.pi)**obs.shape[-1], out=None if cov.requires_grad else norm)
    norm = torch.rsqrt(norm, out=None if cov.requires_grad else norm)

    cov_inv = torch.linalg.inv(cov).unsqueeze(-3)  # (..., n_clu, n_obs, n_var, n_var)
    cent_obs = obs.unsqueeze(-3) - mean.unsqueeze(-2)  # (..., n_clu, n_obs, n_var)
    prob = cent_obs.unsqueeze(-2) @ cov_inv @ cent_obs.unsqueeze(-1)  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob, out=None if prob.requires_grad else prob)

    prob = torch.mul(prob, norm, out=None if prob.requires_grad else prob)
    return prob


def gauss2d(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, *, _check: bool = True
) -> torch.Tensor:
    r"""Compute a 2d gaussian.

    Approximatively 30% faster than general ``gauss``.

    Parameters
    ----------
    obs : torch.Tensor
        The observations of shape (..., \(N\), 2).
        \(\mathbf{x}_i = \begin{pmatrix} x_1 \\ x_2 \\ \end{pmatrix}_j\)
    mean : torch.Tensor
        The 2x1 column mean vector of shape (..., \(K\), 2).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (..., \(K\), 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)

    Returns
    -------
    prob : torch.Tensor
        The prob density to draw the sample obs of shape (..., \(K\), \(N\)).
        It is associated to \(\mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}\).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gauss import gauss2d
    >>> obs = torch.randn((1000, 100, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2))  # (..., n_clu, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>>
    >>> prob = gauss2d(obs, mean, cov)
    >>> prob.shape
    torch.Size([1000, 3, 100])
    >>>
    >>> mean.requires_grad = cov.requires_grad = True
    >>> gauss2d(obs, mean, cov).sum().backward()
    >>>
    """
    if _check:
        check_ingauss(obs, mean, cov)
        assert obs.shape[-1] == 2, f"please use `nd_gauss` for gauss in {obs.shape[-1]} dim"

    norm, inv = inv_cov2d(cov)
    norm = norm.unsqueeze(-1)  # (..., n_clu, 1)
    norm *= (2.0*torch.pi)**2
    norm = torch.rsqrt(norm)  # no "out=None if cov.requires_grad else norm" for jacobian
    inv = inv.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, n_var)

    cent_obs = obs.unsqueeze(-3) - mean.unsqueeze(-2)  # (..., n_clu, n_obs, n_var)
    prob = cent_obs.unsqueeze(-2) @ inv @ cent_obs.unsqueeze(-1)  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob)  # no "out=None if prob.requires_grad else prob" for jacobian

    prob = norm * prob  # no "out=None if prob.requires_grad else prob" for jacobian

    return prob
