#!/usr/bin/env python3

"""Helper for fast linear algebra for 2d matrix."""

import sympy
import torch


def cov2d_to_eigtheta(cov: torch.Tensor, eig: bool=True, theta: bool=True) -> torch.Tensor:
    r"""Rotate the covariance matrix into canonical base, like a PCA.

    Work only for dimension \(D = 2\), ie 2*2 square matrix.

    Parameters
    ----------
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (..., 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    eig : boolean, default=True
        If True, compute the eigen values, leave the field empty otherwise (faster).
    theta : boolean, default=True
        If True, compute the eigen vector rotation, leave the field empty otherwise (faster).

    Returns
    -------
    eigtheta : torch.Tensor
        The concatenation of the eigenvalues and theta
        \( \left[ \lambda_1, \lambda_2, \theta \right] \) of shape (..., 3) such that:
        \(\begin{cases}
            \lambda_1 >= \lambda_2 > 0 \\
            \theta \in \left]-\frac{\pi}{2}, \frac{\pi}{2}\right] \\
            \mathbf{R} = \begin{pmatrix}
                cos(\theta) & -sin(\theta) \\
                sin(\theta) & cos(\theta) \\
            \end{pmatrix} \\
            \mathbf{D} = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \\ \end{pmatrix} \\
            \mathbf{R} \mathbf{\Sigma} \mathbf{R}^{-1} = \mathbf{D} \\
        \end{cases}\)

        \(\begin{cases}
            tr( \mathbf{\Sigma} ) = tr( \mathbf{R} \mathbf{\Sigma} \mathbf{R}^{-1} )
            = tr(\mathbf{D}) \\
            det( \mathbf{\Sigma} ) = det( \mathbf{R} \mathbf{\Sigma} \mathbf{R}^{-1} )
            = det(\mathbf{D}) \\
            \mathbf{\Sigma} \begin{pmatrix} cos(\theta) \\ sin(\theta) \\ \end{pmatrix}
            = \lambda_1 \begin{pmatrix} cos(\theta) \\ sin(\theta) \\ \end{pmatrix} \\
        \end{cases}\)
        \( \Leftrightarrow \begin{cases}
            \lambda_1 + \lambda_2 = \sigma_1 + \sigma_2 \\
            \lambda_1 \lambda_2 = \sigma_1 \sigma_2 - c^2 \\
            \sigma_1 cos(\theta) + c sin(\theta) = \lambda_1 cos(\theta) \\
        \end{cases}\)
        \( \Leftrightarrow \begin{cases}
            \lambda_1 = \frac{1}{2} \left(
                \sigma_1 + \sigma_2 + \sqrt{(2c)^2 + (\sigma_2 - \sigma_1)^2}
            \right) \\
            \lambda_2 = \frac{1}{2} \left(
                \sigma_1 + \sigma_2 + \sqrt{(2c)^2 - (\sigma_2 - \sigma_1)^2}
            \right) \\
            \theta = tan^{-1}\left(
                \frac{ \sigma_2 - \sigma_1 + \sqrt{(2c)^2 + (\sigma_2 - \sigma_1)^2} }{2c}
            \right) \\
        \end{cases}\)


    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from laueimproc.gmm.linalg import cov2d_to_eigtheta
    >>> obs = torch.randn((1000, 10, 2), dtype=torch.float64)  # (..., n_obs, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> eigtheta = cov2d_to_eigtheta(cov)
    >>>
    >>> # check resultst are corrects
    >>> torch.allclose(torch.linalg.eigvalsh(cov).flip(-1), eigtheta[..., :2])
    True
    >>> theta = eigtheta[..., 2]
    >>> rot = [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    >>> rot = torch.as_tensor(np.array(rot)).movedim((0, -2), (1, -1))
    >>> diag = torch.zeros_like(cov)
    >>> diag[..., 0, 0] = eigtheta[..., 0]
    >>> diag[..., 1, 1] = eigtheta[..., 1]
    >>> torch.allclose(torch.linalg.inv(rot) @ cov @ rot, diag)
    True
    >>>
    >>> # check is differenciable
    >>> cov.requires_grad = True
    >>> cov2d_to_eigtheta(cov).sum().backward()
    >>>
    >>> # timing
    >>> import timeit
    >>> min(timeit.repeat(lambda: cov2d_to_eigtheta(cov), repeat=100, number=100))
    0.0349183059988718
    >>> min(timeit.repeat(lambda: torch.linalg.eigh(cov), repeat=100, number=100))
    0.19978590199752944
    >>> min(timeit.repeat(lambda: cov2d_to_eigtheta(cov, eig=False), repeat=100, number=100))
    0.024473870002111653
    >>> min(timeit.repeat(lambda: torch.linalg.eigvalsh(cov), repeat=100, number=100))
    0.07869026600019424
    >>>
    """
    assert isinstance(cov, torch.Tensor), cov.__class__.__name__
    assert cov.shape[-2:] == (2, 2), cov.shape
    assert isinstance(eig, bool), eig.__class__.__name__
    assert isinstance(theta, bool), theta.__class__.__name__

    # preparation
    out = torch.empty((*cov.shape[:-2], 3), dtype=cov.dtype, device=cov.device)
    sigma1 = cov[..., 0, 0]
    sigma2 = cov[..., 1, 1]
    corr = cov[..., 0, 1]

    # calculus
    two_corr = 2 * corr
    sigma_sum = sigma1 + sigma2
    sigma_dif = sigma2 - sigma1
    delta = two_corr**2 + sigma_dif**2
    delta = torch.sqrt(delta, out=None if cov.requires_grad else delta)
    if eig:
        lambda1 = sigma_sum + delta
        lambda2 = sigma_sum - delta
        lambda1 *= 0.5
        lambda2 *= 0.5
        out[..., 0] = lambda1
        out[..., 1] = lambda2
    if theta:
        theta = sigma_dif + delta
        theta /= two_corr
        theta = torch.atan(theta, out=None if cov.requires_grad else theta)
        out[..., 2] = theta
    return out


def _inv_cov2d(cov: torch.Tensor, inv: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper for ``inv_cov2d``."""
    # preparation
    sigma1 = cov[..., 0, 0]
    sigma2 = cov[..., 1, 1]
    corr = cov[..., 0, 1]
    out = torch.empty_like(cov)  # not None for compilation

    # calculus
    det = sigma1*sigma2
    det -= corr**2
    if inv:
        inv_det = 1.0 / det
        out[..., 0, 0] = inv_det * sigma2
        out[..., 1, 1] = inv_det * sigma1
        out[..., 0, 1] = out[..., 1, 0] = -inv_det * corr
    return det, out


def inv_cov2d(cov: torch.Tensor, inv: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the det and the inverse of covariance matrix.

    Parameters
    ----------
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (..., 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    inv : boolean, default=True
        If True, compute the inverse matrix, return empty tensor overwise otherwise (faster).

    Returns
    -------
    det : torch.Tensor
        The determinant of the matrix of shape (...,).
    inv : torch.Tensor
        The inverse matrix of shape (..., 2, 2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.linalg import inv_cov2d
    >>> obs = torch.randn((1000, 10, 2))  # (..., n_obs, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> _, inv = inv_cov2d(cov)
    >>> torch.allclose(torch.linalg.inv(cov), inv)
    True
    >>>
    """
    assert isinstance(cov, torch.Tensor), cov.__class__.__name__
    assert cov.shape[-2:] == (2, 2), cov.shape
    assert isinstance(inv, bool), inv.__class__.__name__

    return _inv_cov2d(cov, inv)


def inv_cov2d_sympy(cov: sympy.Matrix) -> tuple[sympy.Expr, sympy.Matrix]:
    """Same as ``inv_cov2d`` with sympy objects.

    Examples
    --------
    >>> from sympy import *
    >>> from laueimproc.gmm.linalg import inv_cov2d_sympy
    >>> sigma_1, sigma_2 = symbols("sigma_1, sigma_2", real=True, positive=True)
    >>> corr = Symbol("c", real=True)
    >>> cov = Matrix([[sigma_1, corr], [corr, sigma_2]])
    >>> det, inv = inv_cov2d_sympy(cov)
    >>> det
    -c**2 + sigma_1*sigma_2
    >>> inv
    Matrix([
    [-sigma_2/(c**2 - sigma_1*sigma_2),        c/(c**2 - sigma_1*sigma_2)],
    [       c/(c**2 - sigma_1*sigma_2), -sigma_1/(c**2 - sigma_1*sigma_2)]])
    >>>
    """
    assert isinstance(cov, sympy.Matrix), cov.__class__.__name__
    assert cov.shape == (2, 2)
    return cov.det(), sympy.cancel(cov.inv())
