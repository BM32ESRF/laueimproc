#!/usr/bin/env python3

"""Helper for compute a mixture of multivariate gaussians."""

import typing

import numpy as np
import torch

from .check import check_gmm
from .gauss import gauss2d
from laueimproc.opti.rois import rawshapes2rois


def _gmm2d_and_jac_autodiff(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, eta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Automatic torch differenciation version of gmm2d_and_jac."""
    *batch, n_obs, _ = obs.shape
    *_, n_clu, _, _ = mean.shape

    obs = obs.reshape(-1, n_obs, 2)  # (b, n_obs, 2)
    mean = mean.reshape(-1, n_clu, 2, 1)  # (b, n_clu, 2, 1)
    cov = cov.reshape(-1, n_clu, 2, 2)  # (b, n_clu, 2, 2)
    eta = eta.reshape(-1, n_clu)  # (b, n_clu)

    prob = gmm2d(obs, mean, cov, eta, _check=False)
    mean_jac, cov_jac, eta_jac = (
        torch.func.vmap(torch.func.jacfwd(
            lambda o, m, c, e: gmm2d(o, m, c, e, _check=False),
            argnums=(1, 2, 3),
        ))(obs, mean, cov, eta)
    )
    cov_jac[..., 1, 0] = cov_jac[..., 0, 1]  # because only cov_jac[..., 0, 1] is used for gauss

    prob = prob.reshape(*batch, n_obs)
    mean_jac = mean_jac.reshape(*batch, n_obs, n_clu, 2, 1)
    cov_jac = cov_jac.reshape(*batch, n_obs, n_clu, 2, 2)
    eta_jac = eta_jac.reshape(*batch, n_obs, n_clu)

    return prob, mean_jac, cov_jac, eta_jac


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


def cost_and_grad(
    rois: torch.Tensor, shapes: torch.Tensor,
    loss: typing.Union[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    mean: torch.Tensor, cov: torch.Tensor, eta: torch.Tensor,
    *, _check: bool = True, _autodiff: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the grad of the loss between the predicted gmm and the rois.

    Parameters
    ----------
    rois : torch.Tensor
        The unfolded and padded rois of dtype torch.float32 and shape (n, h, w).
    shapes : torch.Tensor
        Contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.
    loss : str callable
        Reduce the rois and the predected rois into a single vector of scalar.
        The shapes are [Tensor(n, h, w), Tensor(n, h, w)] -> Tensor(n,).
        Could be the string "mse".
    mean : torch.Tensor
        The 2x1 column mean vector of shape (n, \(K\), 2, 1).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (n, \(K\), 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    eta : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (n, \(K\)).

    Returns
    -------
    cost : torch.Tensor
        The value of the reduced loss function evaluated for each roi.
        The shape in (n,).
    mean_grad : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (n, \(K\), 2, 1).
    cov_grad : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (n, \(K\), 2, 2).
        Take care of the factor 2 of the coefficient of correlation.
    eta_grad : torch.Tensor
        The gradient of the scalar mass of each gaussian of shape (n, \(K\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import cost_and_grad
    >>> rois = torch.rand((1000, 10, 10))
    >>> loss = lambda pred, roi: (pred - roi)**2
    >>> mean = torch.randn((1000, 3, 2, 1)) + 5.0  # (n, n_clu, n_var, 1)
    >>> cov = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 1, 2, 2).expand(1000, 3, 2, 2).clone()
    >>> eta = torch.rand((1000, 3))  # (n, n_clu)
    >>> eta /= eta.sum(dim=-1, keepdim=True)
    >>>
    >>> cost, mean_grad, cov_grad, eta_grad = cost_and_grad(rois, "mse", mean, cov, eta)
    >>> cost.shape
    torch.Size([1000])
    >>> mean_grad.shape
    torch.Size([1000, 3, 2, 1])
    >>> cov_grad.shape
    torch.Size([1000, 3, 2, 2])
    >>> eta_grad.shape
    torch.Size([1000, 3])
    >>>
    >>> cost_, mean_grad_, cov_grad_, eta_grad_ = cost_and_grad(rois, loss, mean, cov, eta)
    >>> assert torch.allclose(cost, cost_)
    >>> assert torch.allclose(mean_grad, mean_grad_)
    >>> assert torch.allclose(cov_grad, cov_grad_)
    >>> assert torch.allclose(eta_grad, eta_grad_)
    >>>
    >>> import timeit
    >>> min(  # doctest: +SKIP
    ...     timeit.repeat(lambda: cost_and_grad(rois, "mse", mean, cov, eta), repeat=10, number=10)
    ... )
    1.1117540699997335
    >>>
    """
    if _check:
        if isinstance(loss, str):
            assert loss in {"mse"}
        else:
            assert callable(loss), loss.__class__.__name__
        assert isinstance(rois, torch.Tensor), rois.__class__.__name__
        assert rois.ndim == 3, rois.shape
        assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__
        assert shapes.ndim == 2, shapes.shape

    # preparation
    areas = torch.prod(shapes, dim=1).to(dtype=rois.dtype, device=rois.device)
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(1), points_j.unsqueeze(1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n, n_obs, n_var)

    # evaluation of jacobian and loss
    rois_pred, mean_jac, cov_jac, mag_jac = gmm2d_and_jac(
        obs, mean, cov, eta, _check=_check, _autodiff=_autodiff
    )

    # compute cost and cost grad
    if loss == "mse":
        rois_pred -= rois.reshape(*rois_pred.shape)
        cost = torch.sum(rois_pred * rois_pred, dim=1) / areas
        rois_pred += rois_pred  # grad
        rois_pred /= areas.unsqueeze(1)
    else:
        rois_pred = rois_pred.detach()
        rois_pred.requires_grad = True
        rois_pred.grad = None
        cost = loss(rois.reshape(*rois_pred.shape), rois_pred).sum(dim=1) / areas
        cost.sum().backward()
        rois_pred = rois_pred.grad  # grad

    # compute grad
    mean_jac *= rois_pred.reshape(*rois_pred.shape, 1, 1, 1)
    cov_jac *= rois_pred.reshape(*rois_pred.shape, 1, 1, 1)
    mag_jac *= rois_pred.reshape(*rois_pred.shape, 1)
    return cost, torch.sum(mean_jac, dim=1), torch.sum(cov_jac, dim=1), torch.sum(mag_jac, dim=1)


def gmm2d_and_jac(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, eta: torch.Tensor,
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
    prob : torch.Tensor
        Returned value from ``gmm2d`` of shape (..., \(N\)).
    mean_jac : torch.Tensor
        The jacobian of the 2x1 column mean vector of shape (..., \(N\), \(K\), 2, 1).
    cov_jac : torch.Tensor
        The jacobian of the 2x2 half covariance matrix of shape (..., \(N\), \(K\), 2, 2).
        Take care of the factor 2 of the coefficient of correlation.
    eta_jac : torch.Tensor
        The jacobian of the scalar mass of each gaussian of shape (..., \(N\), \(K\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import gmm2d_and_jac
    >>> obs = torch.randn((1000, 100, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2, 1))  # (..., n_clu, n_var, 1)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2).clone()  # (..., n_clu, n_var, n_var)
    >>> eta = torch.rand((1000, 3))  # (..., n_clu)
    >>> eta /= eta.sum(dim=-1, keepdim=True)
    >>>
    >>> prob, mean_jac, cov_jac, eta_jac = gmm2d_and_jac(obs, mean, cov, eta)
    >>> prob.shape
    torch.Size([1000, 100])
    >>> mean_jac.shape
    torch.Size([1000, 100, 3, 2, 1])
    >>> cov_jac.shape
    torch.Size([1000, 100, 3, 2, 2])
    >>> eta_jac.shape
    torch.Size([1000, 100, 3])
    >>>
    >>> prob_, mean_jac_, cov_jac_, eta_jac_ = gmm2d_and_jac(obs, mean, cov, eta, _autodiff=True)
    >>> assert torch.allclose(prob, prob_)
    >>> assert torch.allclose(mean_jac, mean_jac_)
    >>> assert torch.allclose(cov_jac, cov_jac_)
    >>> assert torch.allclose(eta_jac, eta_jac_)
    >>>
    """
    if _check:
        check_gmm((mean, cov, eta))

    *batch, n_obs, _ = obs.shape
    *_, n_clu, _, _ = mean.shape

    if _autodiff:
        return _gmm2d_and_jac_autodiff(obs, mean, cov, eta)

    corr = cov[..., :, 0, 1].unsqueeze(-2)  # (..., 1, n_clu)
    sigma_d0 = cov[..., :, 0, 0].unsqueeze(-2)  # (..., 1, n_clu)
    sigma_d1 = cov[..., :, 1, 1].unsqueeze(-2)  # (..., 1, n_clu)
    obs = obs.unsqueeze(-2) - mean.squeeze(-1).unsqueeze(-3)  # (..., n_obs, n_clu, 2)
    eta = eta.unsqueeze(-2)  # (..., 1, n_clu)
    o_d0 = obs[..., :, :, 0]  # (..., n_obs, n_clu)
    o_d1 = obs[..., :, :, 1]  # (..., n_obs, n_clu)

    _0 = -corr * corr
    _6 = sigma_d0 * sigma_d1
    _2 = _6 + _0  # det
    _3 = torch.rsqrt(_2)
    _1 = -_2**-1  # x**-1 faster than 1/x
    _9 = _1 * o_d0
    dcorr = _1 * o_d1
    dmd0 = dcorr * corr
    dmd0 -= _9 * sigma_d1
    dmd1 = _9 * corr
    dmd1 -= dcorr * sigma_d0
    dsd1 = -0.5 * dmd1 * o_d1
    _13 = -0.5 * dmd0 * o_d0
    dsd1 += _13
    dsd1 = torch.exp(dsd1, out=dsd1)
    _13 = 0.1591549430918953357688838 * dsd1
    deta = _13 * _3
    probs = deta * eta
    _2 **= 3
    _2 = torch.rsqrt(_2, out=_2)
    _2 *= eta
    _16 = 0.07957747154594766788444188 * dsd1 * _2
    _1 *= _1
    _17 = _1 * o_d0
    _18 = _1 * o_d1
    _19 = -0.5 * o_d0
    _20 = -0.5 * o_d1
    _21 = corr * sigma_d0
    _4 = corr * sigma_d1
    dmd0 *= probs
    dmd1 *= probs
    dsd0 = -_16 * sigma_d1
    _23 = _18 * _4
    _24 = sigma_d1 * sigma_d1
    _23 -= _17 * _24
    _23 *= _19
    _1 *= corr
    o_d0 *= _1 * sigma_d1  # ok because o_d0 is a copy
    o_d1 *= _1 * sigma_d0
    _25 = o_d0 - (_18 * _6) - dcorr
    _25 *= _20
    _23 += _25
    _23 *= _13 * _3 * eta
    dsd0 += _23
    _16 *= -sigma_d0
    _23 = o_d1 - (_17 * _6) - _9
    _23 *= _19
    _25 = _17 * _21
    _25 -= sigma_d0 * sigma_d0 * _18
    _25 *= _20
    _23 += _25
    dsd1 *= 0.1591549430918953357688838 * _23 * _3 * eta
    dsd1 += _16
    _0 += _0
    dcorr += (_0 * _18) + (2 * _17 * _4)
    _9 += (_0 * _17) + (2 * _18 * _21)
    dcorr *= _19
    _9 *= _20
    dcorr += _9
    dcorr *= probs
    _9 = _13 * _2 * corr
    dcorr += _9

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
    """Find the algebrical expression of the jacobian."""
    # from laueimproc.gmm.gmm import gmm2d_and_jac_sympy
    # jac = gmm2d_and_jac_sympy()
    from sympy import cancel, exp, pi, pprint, sqrt, symbols, Matrix, Symbol

    o_d0, o_d1 = symbols("o_d0 o_d1", real=True)
    obs = Matrix([[o_d0], [o_d1]])
    mu_d0, mu_d1 = symbols("mu_d0 mu_d1", real=True)
    mean = Matrix([[mu_d0], [mu_d1]])
    sigma_d0, sigma_d1 = symbols("sigma_d0 sigma_d1", real=True, positive=True)
    corr = Symbol("corr", real=True)
    cov = Matrix([[sigma_d0, corr], [corr, sigma_d1]])

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
    # lamb = Lambdify(  # for C
    #     jac,
    #     cst_args={eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr},
    #     shapes={(o_d0, o_d1, eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
    # )
    lamb = Lambdify(  # for torch
        jac,
        shapes={(o_d0, o_d1), (eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
    )
    # print(lamb)
    # print(lamb._tree["dyn_code"])

    return jac
