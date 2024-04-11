#!/usr/bin/env python3

"""Helper for compute a mixture of multivariate gaussians."""

import torch

from .check import check_gmm
from .gauss import gauss2d


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


def _gmm2d_and_jac_autodiff(
    obs: torch.Tensor, mean: torch.Tensor, half_cov: torch.Tensor, eta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Automatic torch differenciation version of gmm2d_and_jac."""
    *batch, n_obs, _ = obs.shape
    *_, n_clu, _, _ = mean.shape

    obs = obs.reshape(-1, n_obs, 2)  # (b, n_obs, 2)
    mean = mean.reshape(-1, n_clu, 2, 1)  # (b, n_clu, 2, 1)
    half_cov = half_cov.reshape(-1, n_clu, 2, 2)  # (b, n_clu, 2, 1)
    eta = eta.reshape(-1, n_clu)  # (b, n_clu)

    prob = gmm2d(obs, mean, half_cov + half_cov.mT, eta, _check=False)
    mean_jac, half_cov_jac, eta_jac = (
        torch.func.vmap(torch.func.jacfwd(
            lambda o, m, hv, e: gmm2d(o, m, hv+hv.mT, e, _check=False),
            argnums=(1, 2, 3),
        ))(obs, mean, half_cov, eta)
    )

    prob = prob.reshape(*batch, n_obs)
    mean_jac = mean_jac.reshape(*batch, n_obs, n_clu, 2, 1)
    half_cov_jac = half_cov_jac.reshape(*batch, n_obs, n_clu, 2, 2)
    eta_jac = eta_jac.reshape(*batch, n_obs, n_clu)

    return prob, mean_jac, half_cov_jac, eta_jac


def gmm2d_and_jac(
    obs: torch.Tensor, mean: torch.Tensor, half_cov: torch.Tensor, eta: torch.Tensor,
    *, _check: bool = True, _autodiff: bool = False
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
    >>> prob_, mean_jac_, half_cov_jac_, eta_jac_ = (
    ...     gmm2d_and_jac(obs, mean, half_cov, eta, _autodiff=True)
    ... )
    >>> assert torch.allclose(prob, prob_)
    >>> assert torch.allclose(mean_jac, mean_jac_)
    >>> assert torch.allclose(half_cov_jac, half_cov_jac_)
    >>> assert torch.allclose(eta_jac, eta_jac_)
    >>>
    """
    if _check:
        check_gmm((mean, half_cov, eta))

    *batch, n_obs, _ = obs.shape
    *_, n_clu, _, _ = mean.shape

    if _autodiff:
        return _gmm2d_and_jac_autodiff(obs, mean, half_cov, eta)

    corr = half_cov[..., :, 0, 1].unsqueeze(-2)  # (..., 1, n_clu)
    sigma_d0 = half_cov[..., :, 0, 0].unsqueeze(-2)  # (..., 1, n_clu)
    sigma_d1 = half_cov[..., :, 1, 1].unsqueeze(-2)  # (..., 1, n_clu)
    mu_d0 = mean[..., :, 0, 0].unsqueeze(-2)  # (..., 1, n_clu)
    mu_d1 = mean[..., :, 1, 0].unsqueeze(-2)  # (..., 1, n_clu)
    eta = eta.unsqueeze(-2)  # (..., 1, n_clu)
    o_d0 = obs[..., :, 0].unsqueeze(-1)  # (..., n_obs, 1)
    o_d1 = obs[..., :, 1].unsqueeze(-1)  # (..., n_obs, 1)

    _0 = corr**2
    _7 = sigma_d0*sigma_d1
    _1 = _0 - _7
    _2 = -_1
    _3 = torch.rsqrt(_2)
    _4 = -mu_d0
    _5 = _4 + o_d0
    _4 = -mu_d1
    _6 = _4 + o_d1
    _4 = _7
    _7 = -2*_7
    _8 = 2*_0
    _7 = _7 + _8
    _7 = 1/_7
    _8 = _7*corr
    _9 = _5*sigma_d1
    _10 = _6*sigma_d0
    _11 = _6*_8
    _12 = -_7*_9
    _11 = _11 + _12
    _11 = -_11*_5/2
    _12 = _5*_8
    _13 = -_10*_7
    _12 = _12 + _13
    _12 = -_12*_6/2
    _11 = _11 + _12
    _11 = torch.exp(_11)
    _12 = 0.07957747154594766788444188*_11
    _13 = _12*_3
    _14 = _13*eta
    _7 = 2.0*_0
    _8 = -2.0*_4  # marker
    _7 = _7 + _8
    _7 = 1/_7
    _15 = _6*_7
    _16 = _5*_7
    _7 = torch.rsqrt(_2)**3
    _7 = _7*eta
    _17 = 0.03978873577297383394222094*_11*_7
    _1 = 1 / (_1**2)
    _18 = 0.5*_5
    _19 = _1*_18
    _20 = 0.5*_6
    _21 = _1*_20
    _0 = _0*_1
    _22 = _15*corr
    _23 = -_16*sigma_d1
    _22 = _22 + _23
    _22 = _14*_22
    _23 = _16*corr
    _24 = -_15*sigma_d0
    _23 = _23 + _24
    _23 = _14*_23
    _24 = -_17*sigma_d1
    _2 = sigma_d1**2
    _25 = -_19*_2
    _26 = _21*corr*sigma_d1
    _25 = _25 + _26
    _25 = -_18*_25
    _26 = -_15
    _27 = -_21*_4
    _28 = 0.5*_1*_5*corr*sigma_d1
    _26 = _26 + _27 + _28
    _26 = -_20*_26
    _25 = _25 + _26
    _2 = 0.07957747154594766788444188 * _11 * _3 * eta
    _25 = _2 * _25
    _24 = _24 + _25
    _17 = -_17*sigma_d0
    _25 = -_16
    _26 = -_19*_4
    _27 = 0.5*_1*_6*corr*sigma_d0
    _25 = _25 + _26 + _27
    _25 = -_18*_25
    _8, _1 = _1, sigma_d0**2
    _21 = -_1*_21
    _19 = _19*corr*sigma_d0
    _19 = _19 + _21
    _19 = -_19*_20
    _19 = _19 + _25
    _11 = _2 * _19
    _11 = _11 + _17
    _17 = -_0*_6
    _9 = _8*_9*corr
    _9 = _15 + _17 + _9
    _9 = -_18*_9
    _15 = -_0*_5
    _10 = _10*_8*corr
    _10 = _10 + _15 + _16
    _10 = -_10*_20
    _9 = _10 + _9
    _9 = _14*_9
    _10 = _12*_7*corr
    _9 = 0.5 * (_10 + _9)

    probs, deta, dmd0, dmd1, dsd0, dsd1, dcorr = _14, _13, _22, _23, _24, _11, _9

    prob = torch.sum(probs, axis=-1)
    mean_jac = torch.cat([
        dmd0.unsqueeze(-1).unsqueeze(-2), dmd1.unsqueeze(-1).unsqueeze(-2)
    ], dim=-2)
    half_cov_jac = torch.cat([
        dsd0.unsqueeze(-1), dcorr.unsqueeze(-1), dcorr.unsqueeze(-1), dsd1.unsqueeze(-1)
    ], dim=-1).reshape(*batch, n_obs, n_clu, 2, 2)
    eta_jac = deta

    return prob, mean_jac, half_cov_jac, eta_jac


def gmm2d_and_jac_sympy():
    """Find the algebrical expression of the jacobian.

    Examples
    --------
    >>> from laueimproc.gmm.gmm import gmm2d_and_jac_sympy
    >>> jac = gmm2d_and_jac_sympy()
    >>>
    """
    from sympy import cancel, exp, pi, pprint, sqrt, symbols, Matrix, Symbol

    o_d0, o_d1 = symbols("o_d0 o_d1", real=True)
    obs = Matrix([[o_d0], [o_d1]])
    mu_d0, mu_d1 = symbols("mu_d0 mu_d1", real=True)
    mean = Matrix([[mu_d0], [mu_d1]])
    sigma_d0, sigma_d1 = symbols("sigma_d0 sigma_d1", real=True, positive=True)
    corr = Symbol("corr", real=True)
    cov = Matrix([[2*sigma_d0, 2*corr], [2*corr, 2*sigma_d1]])

    # compute one gaussian
    det, inv_cov = cov.det(), cancel(cov.inv())
    mean_center = obs - mean
    scalar = (mean_center.T @ inv_cov @ mean_center)[0, 0]
    prob = exp(-scalar/2) / sqrt(4*pi**2*det)
    #print("single gaussian expression:")
    #pprint(prob)

    # compute diff
    eta = Symbol("eta", real=True, positive=True)
    probs = eta * prob
    jac = [
        probs,  # partial, sum on j
        probs.diff(eta),  # equal prob
        probs.diff(mu_d0),
        probs.diff(mu_d1),
        probs.diff(sigma_d0),
        probs.diff(sigma_d1),
        probs.diff(corr),
    ]

    # compile source code
    from cutcutcodec.core.compilation.sympy_to_torch.lambdify import Lambdify
    lamb = Lambdify(  # for C
        jac,
        cst_args={eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr},
        shapes={(o_d0, o_d1, eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
    )
    lamb = Lambdify(  # for torch
        jac,
        shapes={(o_d0, o_d1), (eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
    )
    #print(lamb)

    return jac
