#!/usr/bin/env python3

"""Helper for compute multivariate gaussian."""

import sympy
import torch

from .check import check_ingauss
from .linalg import inv_cov2d, inv_cov2d_sympy


# @torch.compile(fullgraph=True, dynamic=True)  # 50% times faster
def _gauss2d(obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
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
    >>> from laueimproc.gmm.gauss import _gauss2d
    >>> obs = torch.randn((1000, 10, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>>
    >>> prob = _gauss2d(obs, mean, cov)
    >>> prob.shape
    torch.Size([1000, 3, 10])
    >>>
    >>> mean.requires_grad = cov.requires_grad = True
    >>> _gauss2d(obs, mean, cov).sum().backward()
    >>>
    """
    norm, inv = inv_cov2d(cov)
    norm = norm.unsqueeze(-1)  # (..., n_clu, 1)
    norm *= (2.0*torch.pi)**2
    norm = torch.rsqrt(norm)#, out=None if cov.requires_grad else norm)
    inv = inv.unsqueeze(-3) # (..., n_clu, n_obs, n_var, n_var)

    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    prob = cent_obs.transpose(-1, -2) @ inv @ cent_obs  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob)#, out=None if prob.requires_grad else prob)

    prob = torch.mul(prob, norm)#, out=None if prob.requires_grad else prob)
    return prob


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

    if obs.shape[-1] == 2:
        return _gauss2d(obs, mean, cov)

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


def gauss2d_sympy(obs: sympy.Matrix, mean: sympy.Matrix, cov: sympy.Matrix) -> sympy.Expr:
    """Same as ``_gauss2d`` with sympy objects.

    Examples
    --------
    >>> from sympy import *
    >>> from laueimproc.gmm.gauss import gauss2d_sympy
    >>> o_1, o_2 = symbols("o_1, o_2", real=True)
    >>> obs = Matrix([[o_1], [o_2]])
    >>> mu_1, mu_2 = symbols("mu_1, mu_2", real=True)
    >>> mean = Matrix([[mu_1], [mu_2]])
    >>> sigma_1, sigma_2 = symbols("sigma_1, sigma_2", real=True, positive=True)
    >>> corr = Symbol("c", real=True)
    >>> cov = Matrix([[sigma_1, corr], [corr, sigma_2]])
    >>> prob = gauss2d_sympy(obs, mean, cov)
    >>> prob.subs({sigma_1: 1, sigma_2: 2, corr: -1, mu_1: 0, mu_2: 0})
    exp(-o_1*(2*o_1 + o_2)/2 - o_2*(o_1 + o_2)/2)/(2*pi)
    >>>
    """
    assert isinstance(obs, sympy.Matrix), obs.__class__.__name__
    assert obs.shape == (2, 1)
    assert isinstance(cov, sympy.Matrix), cov.__class__.__name__
    assert mean.shape == (2, 1)
    assert isinstance(mean, sympy.Matrix), mean.__class__.__name__
    assert cov.shape == (2, 2)

    det, inv_cov = inv_cov2d_sympy(cov)
    mean_center = obs - mean
    scalar = (mean_center.T @ inv_cov @ mean_center)[0, 0]
    prob = sympy.exp(-scalar/2) / sympy.sqrt(4*sympy.pi**2*det)
    return prob
