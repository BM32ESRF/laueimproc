#!/usr/bin/env python3

"""Helper for fast linear algebra for 2d matrix."""

import torch


def batched_matmul(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Perform a matrix product on the last 2 dimensions.

    Parameters
    ----------
    mat1 : torch.Tensor
        Matrix of shape (..., n, m).
    mat2 : torch.Tensor
        Matrix of shape (..., m, p).

    Returns
    -------
    prod : torch.Tensor
        The matrix of shape (..., n, p).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.linalg import batched_matmul
    >>> mat1 = torch.randn((9, 1, 7, 6, 3, 4))
    >>> mat2 = torch.randn((8, 1, 6, 4, 5))
    >>> batched_matmul(mat1, mat2).shape
    torch.Size([9, 8, 7, 6, 3, 5])
    >>>
    """
    assert isinstance(mat1, torch.Tensor), mat1.__class__.__name__
    assert isinstance(mat2, torch.Tensor), mat2.__class__.__name__
    assert mat1.ndim >= 2 and mat2.ndim >= 2, (mat1.shape, mat2.shape)
    *batch1, n_dim, m_dim = mat1.shape
    *batch2, m_dim, p_dim = mat2.shape
    batch = torch.broadcast_shapes(batch1, batch2)
    batched_mat1 = mat1.expand(*batch, n_dim, m_dim).reshape(-1, n_dim, m_dim)
    batched_mat2 = mat2.expand(*batch, m_dim, p_dim).reshape(-1, m_dim, p_dim)
    batched_out = torch.bmm(batched_mat1, batched_mat2)
    out = batched_out.reshape(*batch, n_dim, p_dim)
    return out


def cov2d_to_eigtheta(cov: torch.Tensor, eig: bool = True, theta: bool = True) -> torch.Tensor:
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
                \sigma_1 + \sigma_2 - \sqrt{(2c)^2 + (\sigma_2 - \sigma_1)^2}
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
    >>> obs = torch.randn((1000, 100, 2), dtype=torch.float64)  # (..., n_obs, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> eigtheta = cov2d_to_eigtheta(cov)
    >>>
    >>> # check resultst are corrects
    >>> torch.allclose(torch.linalg.eigvalsh(cov).flip(-1), eigtheta[..., :2])
    True
    >>> theta = eigtheta[..., 2]
    >>> rot = [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    >>> rot = torch.asarray(np.array(rot)).movedim((0, -2), (1, -1))
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
    >>> def timer():
    ...     a = min(timeit.repeat(lambda: cov2d_to_eigtheta(cov), repeat=10, number=100))
    ...     b = min(timeit.repeat(lambda: torch.linalg.eigh(cov), repeat=10, number=100))
    ...     print(f"torch is {b/a:.2f} times slowler")
    ...     a = min(timeit.repeat(lambda: cov2d_to_eigtheta(cov, eig=False), repeat=10, number=100))
    ...     b = min(timeit.repeat(lambda: torch.linalg.eigvalsh(cov), repeat=10, number=100))
    ...     print(f"torch is {b/a:.2f} times slowler")
    ...
    >>> timer()  # doctest: +SKIP
    torch is 2.84 times slowler
    torch is 3.87 times slowler
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
        theta = torch.where(two_corr != 0, theta/two_corr, torch.where(theta != 0, torch.inf, 0.0))
        theta = torch.atan(theta, out=None if cov.requires_grad else theta)
        out[..., 2] = theta
    return out


def _inv_cov2d(cov: torch.Tensor, inv: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Help ``inv_cov2d``."""
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


def inv_cov2d(cov: torch.Tensor, inv: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
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
    >>> obs = torch.randn((1000, 100, 2))  # (..., n_obs, n_var)
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
