#!/usr/bin/env python3

"""Helper for compute multivariate gaussian."""

import sympy
import torch

from .check import check_ingauss
from .linalg import inv_cov2d, inv_cov2d_sympy


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
    >>> obs = torch.randn((1000, 10, 2))  # (..., n_obs, n_var)
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
    norm = torch.rsqrt(norm, out=None if cov.requires_grad else norm)
    inv = inv.unsqueeze(-3) # (..., n_clu, n_obs, n_var, n_var)

    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    prob = cent_obs.transpose(-1, -2) @ inv @ cent_obs  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob, out=None if prob.requires_grad else prob)

    prob = torch.mul(prob, norm, out=None if prob.requires_grad else prob)
    return prob


def gauss2d_and_grad(
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
        Returned value from ``gauss2d``.
    mean_grad : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (..., \(K\), 2, 1).
    half_cov_grad : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (..., \(K\), 2, 2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gauss import gauss2d_and_grad
    >>> obs = torch.randn((1000, 10, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  # (..., n_clu, n_var, n_var)
    >>> half_cov = cov / 2
    >>>
    >>> prob, grad_mean, grad_half_cov = gauss2d_and_grad(obs, mean, half_cov)
    >>> prob.shape
    torch.Size([1000, 3, 10])
    >>> grad_mean.shape
    torch.Size([1000, 3, 2, 1])
    >>> grad_half_cov.shape
    torch.Size([1000, 3, 2, 2])
    >>>
    """
    if _check:
        assert isinstance(mean, torch.Tensor), mean.__class__.__name__
        assert isinstance(half_cov, torch.Tensor), half_cov.__class__.__name__
        assert half_cov.shape[-2:] == (2, 2), half_cov.shape

    mean_, half_cov_ = mean.detach().clone(), half_cov.detach().clone()
    mean_.requires_grad = half_cov_.requires_grad = True
    mean_.grad = half_cov_.grad = None
    prob = gauss2d(obs, mean_, half_cov_ + half_cov_.mT, _check=_check)
    prob.sum().backward()
    if obs.requires_grad or mean.requires_grad or half_cov.requires_grad:
        prob = gauss2d(obs, mean, half_cov + half_cov.mT, _check=False)
    return prob, mean_.grad, half_cov_.grad


def gauss2d_sympy(obs: sympy.Matrix, mean: sympy.Matrix, cov: sympy.Matrix) -> sympy.Expr:
    """Same as ``_gauss2d`` with sympy objects.

    Examples
    --------
    >>> from sympy import *
    >>> from laueimproc.gmm.gauss import gauss2d_sympy
    >>> o_i, o_j = symbols("o_i o_j", real=True)
    >>> obs = Matrix([[o_i], [o_j]])
    >>> mu_1, mu_2 = symbols("mu_1 mu_2", real=True)
    >>> mean = Matrix([[mu_1], [mu_2]])
    >>> sigma_1, sigma_2 = symbols("sigma_1 sigma_2", real=True, positive=True)
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


def gauss2dgrad_sympy(obs: sympy.Matrix, mean: sympy.Matrix, cov: sympy.Matrix):
    """Same as ``gauss2d_sympy`` with the diff of [mu_1, mu_2, sigma_1, sigma_22, corr].

    Examples
    --------
    >>> from sympy import *
    >>> from laueimproc.gmm.gauss import gauss2dgrad_sympy
    >>> o_i, o_j = symbols("o_i o_j", real=True)
    >>> obs = Matrix([[o_i], [o_j]])
    >>> mu_1, mu_2 = symbols("mu_1 mu_2", real=True)
    >>> mean = Matrix([[mu_1], [mu_2]])
    >>> sigma_1, sigma_2 = symbols("sigma_1 sigma_2", real=True, positive=True)
    >>> corr = Symbol("c", real=True)
    >>> cov = Matrix([[sigma_1, corr], [corr, sigma_2]])
    >>> probgrad = gauss2dgrad_sympy(obs, mean, cov)
    >>> for k, v in cse(probgrad)[0]:
    ...     print(f"{k} = {v}")
    ...
    x0 = c**2
    x1 = -sigma_1*sigma_2 + x0
    x2 = -x1
    x3 = 1/sqrt(x2)
    x4 = 1/pi
    x5 = -mu_2 + o_j
    x6 = 1/x1
    x7 = x5*x6
    x8 = -mu_1 + o_i
    x9 = x6*x8
    x10 = c*x7 - sigma_2*x9
    x11 = x8/2
    x12 = c*x9 - sigma_1*x7
    x13 = x5/2
    x14 = exp(-x10*x11 - x12*x13)
    x15 = x14*x4
    x16 = x15/2
    x17 = x16*x3
    x18 = x2**(-3/2)
    x19 = x15*x18/4
    x20 = x1**(-2)
    x21 = x20*x5
    x22 = c*sigma_2
    x23 = x20*x8
    x24 = sigma_1*sigma_2
    x25 = c*sigma_1
    x26 = 2*x0
    >>> for k, v in zip(("prob", "dm1", "dm2", "ds1", "ds2", "dc"), cse(probgrad)[1]):
    ...     print(f"{k} = {v}")
    ...
    prob = x17
    dm1 = x10*x17
    dm2 = x12*x17
    ds1 = -sigma_2*x19 + x14*x3*x4*(-x11*(-sigma_2**2*x23 + x21*x22) - x13*(c*sigma_2*x20*x8 - x21*x24 - x7))/2
    ds2 = -sigma_1*x19 + x14*x3*x4*(-x11*(c*sigma_1*x20*x5 - x23*x24 - x9) - x13*(-sigma_1**2*x21 + x23*x25))/2
    dc = c*x16*x18 + x17*(-x11*(-x21*x26 + 2*x22*x23 + x7) - x13*(2*x21*x25 - x23*x26 + x9))
    >>>
    """
    prob = gauss2d_sympy(obs, mean, cov)
    return [
        prob,
        prob.diff(mean[0, 0]), prob.diff(mean[1, 0]),
        prob.diff(cov[0, 0]), prob.diff(cov[1, 1]), prob.diff(cov[0, 1])
    ]
    # from cutcutcodec.core.compilation.sympy_to_torch.lambdify import Lambdify
    # Lambdify(
    #     probgrad,
    #     cst_args={mu_1, mu_2, sigma_1, sigma_2, corr},
    #     shapes={(o_i, o_j, mu_1, mu_2, sigma_1, sigma_2, corr)}
    # )
