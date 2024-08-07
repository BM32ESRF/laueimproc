#!/usr/bin/env python3

"""Helper for compute a mixture of multivariate gaussians."""

import logging

import torch

try:
    from laueimproc.gmm import c_gmm
except ImportError:
    logging.warning(
        "failed to import laueimproc.gmm.c_gmm, a slow python version is used instead"
    )
    c_gmm = None
from laueimproc.opti.rois import rawshapes2rois
from .check import check_gmm
from .gauss import gauss2d


def _gmm2d_and_jac_autodiff(
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, eta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Automatic torch differenciation version of gmm2d_and_jac."""
    *batch, n_obs, _ = obs.shape
    *_, n_clu, _ = mean.shape

    obs = obs.reshape(-1, n_obs, 2)  # (b, n_obs, 2)
    mean = mean.reshape(-1, n_clu, 2)  # (b, n_clu, 2)
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
    mean_jac = mean_jac.reshape(*batch, n_obs, n_clu, 2)
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
        The 2x1 column mean vector of shape (..., \(K\), 2).
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


def gmm2d_and_jac(  # pylint: disable=R0914
    obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, mag: torch.Tensor,
    *, _check: bool = True, _autodiff: bool = False, **_
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the grad of a 2d mixture gaussian model.

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
    mag : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (..., \(K\)).

    Returns
    -------
    prob : torch.Tensor
        Returned value from ``gmm2d`` of shape (..., \(N\)).
    mean_jac : torch.Tensor
        The jacobian of the 2x1 column mean vector of shape (..., \(N\), \(K\), 2).
    cov_jac : torch.Tensor
        The jacobian of the 2x2 half covariance matrix of shape (..., \(N\), \(K\), 2, 2).
        Take care of the factor 2 of the coefficient of correlation.
    mag_jac : torch.Tensor
        The jacobian of the scalar mass of each gaussian of shape (..., \(N\), \(K\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import gmm2d_and_jac
    >>> obs = torch.randn((1000, 100, 2))  # (..., n_obs, n_var)
    >>> mean = torch.randn((1000, 3, 2))  # (..., n_clu, n_var)
    >>> cov = obs.mT @ obs  # create real symmetric positive covariance matrix
    >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2).clone()  # (..., n_clu, n_var, n_var)
    >>> mag = torch.rand((1000, 3))  # (..., n_clu)
    >>> mag /= mag.sum(dim=-1, keepdim=True)
    >>>
    >>> prob, mean_jac, cov_jac, mag_jac = gmm2d_and_jac(obs, mean, cov, mag)
    >>> prob.shape
    torch.Size([1000, 100])
    >>> mean_jac.shape
    torch.Size([1000, 100, 3, 2])
    >>> cov_jac.shape
    torch.Size([1000, 100, 3, 2, 2])
    >>> mag_jac.shape
    torch.Size([1000, 100, 3])
    >>>
    >>> prob_, mean_jac_, cov_jac_, mag_jac_ = gmm2d_and_jac(obs, mean, cov, mag, _autodiff=True)
    >>> assert torch.allclose(prob, prob_)
    >>> assert torch.allclose(mean_jac, mean_jac_)
    >>> assert torch.allclose(cov_jac, cov_jac_)
    >>> assert torch.allclose(mag_jac, mag_jac_)
    >>>
    """
    if _check:
        check_gmm((mean, cov, mag))

    if _autodiff:
        return _gmm2d_and_jac_autodiff(obs, mean, cov, mag)

    corr = cov[..., 0, 1].unsqueeze(-2).unsqueeze(-1)  # (..., 1, n_clu, 1)
    sigmas = cov[..., [1, 0], [1, 0]].unsqueeze(-3)  # (..., 1, n_clu, 2)
    obs = obs.unsqueeze(-2) - mean.unsqueeze(-3)  # (..., n_obs, n_clu, 2)
    mag = mag.unsqueeze(-2).unsqueeze(-1)  # (..., 1, n_clu, 1)

    dcorr = corr**2  # (..., 1, n_clu, 1)
    x25 = sigmas.prod(-1, keepdim=True)  # (..., 1, n_clu, 1)
    x1 = x25 - dcorr  # (..., 1, n_clu, 1)
    x3 = torch.rsqrt(x1)  # (..., 1, n_clu, 1)
    x6 = x1**-1  # (..., 1, n_clu, 1), x**-1 faster than 1/x
    x7 = x6 * obs  # (..., n_obs, n_clu, 2)
    obs *= 0.5  # ok because copy
    mean_jac = sigmas*x7 - corr*x7.flip(-1)  # (..., n_obs, n_clu, 2)
    x14 = torch.exp(-(mean_jac*obs).sum(-1, keepdim=True)) / (2*torch.pi)  # (..., n_obs, n_clu, 1)
    mag_jac = x14 * x3  # (..., n_obs, n_clu, 1)
    prob = mag * mag_jac  # (..., n_obs, n_clu, 1)
    mean_jac *= prob  # (..., n_obs, n_clu, 2)
    x14 *= mag
    x2 = x3**3 * x14  # (..., n_obs, n_clu, 1)
    x6 *= x6  # (..., 1, n_clu, 1)
    x22 = 2 * x6 * obs  # (..., n_obs, n_clu, 2)
    x22f = x22.flip(-1)
    x23 = corr * sigmas  # (..., 1, n_clu, 2)
    dcorr += dcorr  # (..., 1, n_clu, 1)
    obsf = obs.flip(-1)
    dcorr = corr*x2 - prob*(obsf*(2*x23.flip(-1)*x22f - x22*dcorr - x7)).sum(-1, keepdim=True)
    x3 = -x14 * x3 * (  # (..., n_obs, n_clu, 2)
        obs*(x23*x22f - sigmas**2*x22) + obsf*(2*x6*x23*obs - x22f*x25 + x7.flip(-1))
    ) - 0.5*sigmas*x2
    prob = torch.sum(prob, axis=-2).squeeze(-1)
    cov_jac = torch.cat([
        x3[..., 0].unsqueeze(-1), dcorr, dcorr, x3[..., 1].unsqueeze(-1)
    ], dim=-1).reshape(*obs.shape, 2)

    return prob, mean_jac, cov_jac, mag_jac.squeeze(-1)


def mse_cost(
    data: bytearray, bboxes: torch.Tensor,
    mean: torch.Tensor, cov: torch.Tensor, mag: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    r"""Compute the mse loss between the predicted gmm and the rois.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.
    mean : torch.Tensor
        The 2x1 column mean vector of shape (n, \(K\), 2).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (n, \(K\), 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    mag : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (n, \(K\)).

    Returns
    -------
    cost : torch.Tensor
        The value of the reduced loss function evaluated for each roi.
        The shape in (n,).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import mse_cost
    >>> from laueimproc.opti.rois import roisshapes2raw
    >>> rois = torch.rand((1000, 10, 10))
    >>> bboxes = torch.full((1000, 4), 10, dtype=torch.int16)
    >>> mean = torch.randn((1000, 3, 2)) + 15.0  # (n, n_clu, n_var, 1)
    >>> cov = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 1, 2, 2).expand(1000, 3, 2, 2).clone()
    >>> mag = torch.rand((1000, 3))  # (n, n_clu)
    >>> mag /= mag.sum(dim=-1, keepdim=True)
    >>> data = roisshapes2raw(rois, bboxes[:, 2:])
    >>>
    >>> cost = mse_cost(data, bboxes, mean, cov, mag)
    >>> cost.shape
    torch.Size([1000])
    >>>
    >>> cost_ = mse_cost(data, bboxes, mean, cov, mag, _no_c=True)
    >>> assert torch.allclose(cost, cost_)
    >>>
    """
    if not kwargs.get("_no_c", False) and c_gmm is not None:
        return torch.from_numpy(
            c_gmm.mse_cost(
                data,
                bboxes.numpy(force=True),
                mean.numpy(force=True),
                cov.numpy(force=True),
                mag.numpy(force=True),
            )
        ).to(bboxes.device)

    # verifications
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    # preparation
    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=kwargs.get("_no_c", False))
    obs = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    obs = (obs[0].ravel(), obs[1].ravel())
    obs = torch.cat([obs[0].unsqueeze(1), obs[1].unsqueeze(1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1).clone()  # (n, n_obs, n_var)
    obs += bboxes[:, :2].to(torch.float32).unsqueeze(1)  # rel to abs

    # prediction
    rois_pred = gmm2d(obs, mean, cov, mag)
    rois_pred = rois_pred.reshape(*rois.shape)

    # mse
    costs = [
        float(torch.mean((rois[i] - rois_pred[i])**2))
        for i, (h, w) in enumerate(bboxes[:, 2:].tolist())
    ]
    return torch.asarray(costs, dtype=torch.float32, device=bboxes.device)


def mse_cost_and_grad(
    data: bytearray, bboxes: torch.Tensor,
    mean: torch.Tensor, cov: torch.Tensor, mag: torch.Tensor,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the grad of the loss between the predicted gmm and the rois.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.
    mean : torch.Tensor
        The 2x1 column mean vector of shape (n, \(K\), 2).
        \(\mathbf{\mu}_j = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \end{pmatrix}_j\)
    cov : torch.Tensor
        The 2x2 covariance matrix of shape (n, \(K\), 2, 2).
        \(\mathbf{\Sigma} = \begin{pmatrix} \sigma_1 & c \\ c & \sigma_2 \\ \end{pmatrix}\)
        with \(\begin{cases}
            \sigma_1 > 0 \\
            \sigma_2 > 0 \\
        \end{cases}\)
    mag : torch.Tensor
        The scalar mass of each gaussian \(\eta_j\) of shape (n, \(K\)).

    Returns
    -------
    cost : torch.Tensor
        The value of the reduced loss function evaluated for each roi.
        The shape in (n,).
    mean_grad : torch.Tensor
        The gradient of the 2x1 column mean vector of shape (n, \(K\), 2).
    cov_grad : torch.Tensor
        The gradient of the 2x2 half covariance matrix of shape (n, \(K\), 2, 2).
        Take care of the factor 2 of the coefficient of correlation.
    mag_grad : torch.Tensor
        The gradient of the scalar mass of each gaussian of shape (n, \(K\)).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.gmm.gmm import mse_cost_and_grad
    >>> from laueimproc.opti.rois import roisshapes2raw
    >>> rois = torch.rand((1000, 10, 10))
    >>> bboxes = torch.full((1000, 4), 10, dtype=torch.int16)
    >>> mean = torch.randn((1000, 3, 2)) + 15.0  # (n, n_clu, n_var)
    >>> cov = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 1, 2, 2).expand(1000, 3, 2, 2).clone()
    >>> mag = torch.rand((1000, 3))  # (n, n_clu)
    >>> mag /= mag.sum(dim=-1, keepdim=True)
    >>> data = roisshapes2raw(rois, bboxes[:, 2:])
    >>>
    >>> cost, mean_grad, cov_grad, mag_grad = mse_cost_and_grad(data, bboxes, mean, cov, mag)
    >>> tuple(cost.shape), tuple(mean_grad.shape), tuple(cov_grad.shape), tuple(mag_grad.shape)
    ((1000,), (1000, 3, 2), (1000, 3, 2, 2), (1000, 3))
    >>>
    >>> cost_, mean_g, cov_g, mag_g = mse_cost_and_grad(data, bboxes, mean, cov, mag, _no_c=True)
    >>> assert torch.allclose(cost, cost_)
    >>> assert torch.allclose(mean_grad, mean_g)
    >>> assert torch.allclose(cov_grad, cov_g)
    >>> assert torch.allclose(mag_grad, mag_g)
    >>>
    """
    if not kwargs.get("_no_c", False) and c_gmm is not None:
        return tuple(
            torch.from_numpy(tensor).to(bboxes.device)
            for tensor in c_gmm.mse_cost_and_grad(
                data,
                bboxes.numpy(force=True),
                mean.numpy(force=True),
                cov.numpy(force=True),
                mag.numpy(force=True),
            )
        )

    # verifications
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    # recursive delegation (no 0 padding)
    if bboxes.shape[0] > 1:
        areas = torch.cumsum(bboxes[:, 2].to(torch.int32) * bboxes[:, 3].to(torch.int32), dim=0)
        areas = [0] + areas.tolist()
        grad = zip(
            *(
                mse_cost_and_grad(
                    data[torch.float32.itemsize*bl:torch.float32.itemsize*bh],
                    bboxes[i].unsqueeze(0),
                    mean[i].unsqueeze(0),
                    cov[i].unsqueeze(0),
                    mag[i].unsqueeze(0),
                    **kwargs,
                )
                for i, (bl, bh) in enumerate(zip(areas[:-1], areas[1:]))
            ),
            strict=True,
        )
        grad = tuple(torch.cat(tensors, dim=0) for tensors in grad)
        return grad

    # preparation
    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=kwargs.get("_no_c", False))
    areas = bboxes[:, 2].to(torch.float32) * bboxes[:, 3].to(torch.float32)
    obs = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    obs = (obs[0].ravel(), obs[1].ravel())
    obs = torch.cat([obs[0].unsqueeze(1), obs[1].unsqueeze(1)], dim=1)
    obs = obs.expand(rois.shape[0], -1, -1).clone()  # (n, n_obs, n_var)
    obs += bboxes[:, :2].to(torch.float32).unsqueeze(1)  # rel to abs

    # evaluation of jacobian and loss
    rois_pred, mean_jac, cov_jac, mag_jac = gmm2d_and_jac(obs, mean, cov, mag, **kwargs)

    # compute cost
    rois_pred -= rois.reshape(*rois_pred.shape)
    cost = torch.sum(rois_pred * rois_pred, dim=1) / areas
    rois_pred += rois_pred
    rois_pred /= areas.unsqueeze(1)  # grad of mse

    # compute grad
    mean_jac *= rois_pred.reshape(*rois_pred.shape, 1, 1)
    cov_jac *= rois_pred.reshape(*rois_pred.shape, 1, 1, 1)
    mag_jac *= rois_pred.reshape(*rois_pred.shape, 1)
    return cost, torch.sum(mean_jac, dim=1), torch.sum(cov_jac, dim=1), torch.sum(mag_jac, dim=1)


# def gmm2d_and_jac_sympy():
#     """Find the algebrical expression of the jacobian."""
#     # from laueimproc.gmm.gmm import gmm2d_and_jac_sympy
#     # jac = gmm2d_and_jac_sympy()
#     from sympy import cancel, exp, pi, sqrt, symbols, Matrix, Symbol

#     o_d0, o_d1 = symbols("o_d0 o_d1", real=True)
#     obs = Matrix([[o_d0], [o_d1]])
#     mu_d0, mu_d1 = symbols("mu_d0 mu_d1", real=True)
#     mean = Matrix([[mu_d0], [mu_d1]])
#     sigma_d0, sigma_d1 = symbols("sigma_d0 sigma_d1", real=True, positive=True)
#     corr = Symbol("corr", real=True)
#     cov = Matrix([[sigma_d0, corr], [corr, sigma_d1]])

#     # compute one gaussian
#     det, inv_cov = cov.det(), cancel(cov.inv())
#     mean_center = obs - mean
#     scalar = (mean_center.T @ inv_cov @ mean_center)[0, 0]
#     prob = exp(-scalar/2)# / sqrt(4*pi**2*det)
#     # print("single gaussian expression:")
#     # pprint(prob)

#     # compute diff
#     eta = Symbol("eta", real=True, positive=True)
#     probs = eta * prob
#     jac = [
#         probs,  # partial, sum on j
#         probs.diff(eta),  # equal prob
#         probs.diff(mu_d0),
#         probs.diff(mu_d1),
#         probs.diff(sigma_d0),
#         probs.diff(sigma_d1),
#         probs.diff(corr),
#     ]

#     # pseudo simplification
#     from sympy import cse, pprint
#     pprint(cse(jac, list=False, optimizations="basic", order="canonical"))
#     print()
#     print(cse(jac, list=False, optimizations="basic", order="canonical"))

#     # compile source code
#     from cutcutcodec.core.compilation.sympy_to_torch.lambdify import (
#         Lambdify
#     )
#     # lamb = Lambdify(  # for C
#     #     jac,
#     #     cst_args={eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr},
#     #     shapes={(o_d0, o_d1, eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
#     # )
#     lamb = Lambdify(  # for torch
#         jac,
#         shapes={(o_d0, o_d1), (eta, mu_d0, mu_d1, sigma_d0, sigma_d1, corr)}
#     )
#     print(lamb)
#     print(lamb._tree["dyn_code"])

#     return jac
