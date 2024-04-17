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
    >>> mean = torch.randn((1000, 3, 4, 1))  # (..., n_clu, n_var, 1)
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
    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    prob = cent_obs.transpose(-1, -2) @ cov_inv @ cent_obs  # (..., n_clu, n_obs, 1, 1)
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
        The 2x1 column mean vector of shape (..., \(K\), 2, 1).
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
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>>
    >>> prob = gauss2d(obs, mean, cov)
    >>> prob.shape
    torch.Size([1000, 3, 10])
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
    inv = inv.unsqueeze(-3) # (..., n_clu, n_obs, n_var, n_var)

    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    prob = cent_obs.transpose(-1, -2) @ inv @ cent_obs  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob)  # no "out=None if prob.requires_grad else prob" for jacobian

    prob = norm * prob  # no "out=None if prob.requires_grad else prob" for jacobian
    return prob


def gauss2d_and_jac(
    obs: torch.Tensor, mean: torch.Tensor, half_cov: torch.Tensor, *, _check: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the grad of a 2d gauss.

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

    Returns
    -------
    prob : torch.Tensor
        Returned value from ``gauss2d``, of shape (..., \(K\), \(N\)).
    mean_jac : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (..., \(K\), \(N\), \(K\), 2, 1).
    half_cov_jac : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (..., \(K\), \(N\), \(K\), 2, 2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gauss import gauss2d_and_jac
    >>> obs = torch.randn((1000, 100, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>> half_cov = cov / 2
    >>>
    >>> prob, mean_jac, half_cov_jac = gauss2d_and_jac(obs, mean, half_cov)
    >>> prob.shape
    torch.Size([1000, 3, 10])
    >>> mean_jac.shape
    torch.Size([1000, 3, 10, 3, 2, 1])
    >>> half_cov_jac.shape
    torch.Size([1000, 3, 10, 3, 2, 2])
    >>>
    """
    if _check:
        assert isinstance(mean, torch.Tensor), mean.__class__.__name__
        assert isinstance(half_cov, torch.Tensor), half_cov.__class__.__name__
        assert half_cov.shape[-2:] == (2, 2), half_cov.shape

    *batch, n_obs, _ = obs.shape
    *_, n_clu, _, _ = mean.shape
    obs = obs.reshape(-1, n_obs, 2)  # (b, n_obs, 2)
    mean = mean.reshape(-1, n_clu, 2, 1)  # (b, n_clu, 2, 1)
    half_cov = half_cov.reshape(-1, n_clu, 2, 2)  # (b, n_clu, 2, 1)

    prob = gauss2d(obs, mean, half_cov + half_cov.mT, _check=_check)
    prob = prob.reshape(*batch, n_clu, n_obs)
    mean_jac, half_cov_jac = (
        torch.func.vmap(torch.func.jacfwd(
            lambda o, m, hv: gauss2d(o, m, hv+hv.mT, _check=False),
            argnums=(1, 2),
        ))(obs, mean, half_cov)
    )
    mean_jac = mean_jac.reshape(*batch, n_clu, n_obs, n_clu, 2, 1)
    half_cov_jac = half_cov_jac.reshape(*batch, n_clu, n_obs, n_clu, 2, 2)
    return prob, mean_jac, half_cov_jac
