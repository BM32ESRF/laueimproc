#!/usr/bin/env python3

"""Gaussian mixture model fited by the expequancy maximisation."""

import logging
import math
import numbers
import pickle
import typing

import torch

from laueimproc.classes.tensor import Tensor


def _aic(
    obs: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> Tensor:
    """Akaike Information Criterion.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
        This is an output parameter changed inplace.
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
        This is an output parameter changed inplace.
    eta : Tensor
        The relative mass of each gaussian of shape (..., nbr_clusters).
        This is an output parameter changed inplace.

    Returns
    -------
    aic : Tensor
        A real positive scalar of shape (...,).
        The smaller this criterion, the better the model.

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    free_parameters: int = (
        (obs.shape[-1] * (obs.shape[-1]+1))**2 // 2  # nbr of parameters in cov matrix
        + obs.shape[-1]  # nbr of parameters in mean vector
    ) * eta.shape[-1]  # nbr of gaussians
    aic = _log_likelihood(obs, dup_w, mean, cov, eta)
    aic *= -2.0
    aic += 2 * float(free_parameters)
    return aic


def _bic(
    obs: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> Tensor:
    """Compute the Bayesian Information Criterion.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
        This is an output parameter changed inplace.
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
        This is an output parameter changed inplace.
    eta : Tensor
        The relative mass of each gaussian of shape (..., nbr_clusters).
        This is an output parameter changed inplace.

    Returns
    -------
    bic : Tensor
        A real positive scalar of shape (...,).
        The smaller this criterion, the better the model.

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    free_parameters: int = (
        (obs.shape[-1] * (obs.shape[-1]+1))**2 // 2  # nbr of parameters in cov matrix
        + obs.shape[-1]  # nbr of parameters in mean vector
    ) * eta.shape[-1]  # nbr of gaussians
    if dup_w is None:
        log_n_obs = math.log(obs.shape[-2])
    else:
        log_n_obs = torch.sum(dup_w, axis=-1, keepdim=False)
        log_n_obs = torch.log(log_n_obs, out=log_n_obs)
    bic = _log_likelihood(obs, dup_w, mean, cov, eta)
    bic *= -2.0
    bic += log_n_obs * float(free_parameters)
    return bic


def _likelihood(
    obs: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> Tensor:
    """Compute the likelihood.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
        This is an output parameter changed inplace.
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
        This is an output parameter changed inplace.
    eta : Tensor
        The relative mass of each gaussian of shape (..., nbr_clusters).
        This is an output parameter changed inplace.

    Returns
    -------
    likelihood : Tensor
        A real positive scalar of shape (...,).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    prob = _gauss(obs, mean, cov)  # (..., n_clu, n_obs)
    prob **= dup_w.unsqueeze(-2)
    prob *= eta.unsqueeze(-1)
    ind_prob = torch.sum(prob, axis=-2, keepdim=False)  # (..., n_obs)
    likelihood = torch.prod(ind_prob, axis=-1, keepdim=False)  # (...,)
    return likelihood


def _log_likelihood(
    obs: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> Tensor:
    """Compute the log likelihood.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
        This is an output parameter changed inplace.
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
        This is an output parameter changed inplace.
    eta : Tensor
        The relative mass of each gaussian of shape (..., nbr_clusters).
        This is an output parameter changed inplace.

    Returns
    -------
    log_likelihood : Tensor
        A real positive scalar of shape (...,).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    prob = _gauss(obs, mean, cov)  # (..., n_clu, n_obs)
    prob **= dup_w.unsqueeze(-2)
    prob *= eta.unsqueeze(-1)
    ind_prob = torch.sum(prob, axis=-2, keepdim=False)  # (..., n_obs)
    ind_prob = torch.log(ind_prob, out=None if ind_prob.requires_grad else ind_prob)
    log_likelihood = torch.sum(ind_prob, axis=-1, keepdim=False)  # (...,)
    return log_likelihood


def _gauss(obs: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
    """Compute the density of probabilitie of the normal law at the point obs.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).

    Returns
    -------
    prob : Tensor
        The prob density to draw the sample obs of shape (..., nbr_clusters, nbr_observation).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    cov_inv = torch.linalg.inv(cov).unsqueeze(-3)  # (..., n_clu, n_obs, n_var, n_var)
    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    prob = cent_obs.transpose(-1, -2) @ cov_inv @ cent_obs  # (..., n_clu, n_obs, 1, 1)
    prob = prob.squeeze(-1).squeeze(-1)  # (..., n_clu, n_obs)
    prob *= -.5
    prob = torch.exp(prob, out=None if prob.requires_grad else prob)
    del cov_inv, cent_obs
    norm = torch.linalg.det(cov).unsqueeze(-1)  # (..., n_clu, 1)
    norm *= (2.0*torch.pi)**obs.shape[-1]
    norm = torch.rsqrt(norm, out=norm)
    prob *= norm
    return prob


def _fit_one_cluster(obs: Tensor, std_w: Tensor, dup_w: Tensor) -> tuple[Tensor, Tensor]:
    """Fit one gaussian.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    std_w : Tensor, optional
        The inv std weights of shape (..., nbr_observation).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).

    Returns
    -------
    mean : Tensor
        The column mean vector of shape (..., nbr_variables, 1).
    cov : Tensor
        The covariance matrix of shape (..., nbr_variables, nbr_variables).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    weights_prod = (  # (..., n_obs)
        (1.0 if std_w is None else std_w)
        * (1.0 if dup_w is None else dup_w)
    )
    if std_w is None and dup_w is None:
        mean = torch.mean(obs, axis=-2, keepdim=True)  # (..., 1, n_var)
    else:
        mean = torch.sum((weights_prod.unsqueeze(-1) * obs), axis=-2, keepdim=True)
        mean /= torch.sum(weights_prod.unsqueeze(-1), axis=-2, keepdim=True)  # (..., 1, n_var)
    cent_obs = (obs - mean).unsqueeze(-1)  # (..., n_obs, n_var, 1)
    mean = mean.squeeze(-2).unsqueeze(-1)  # (..., n_var, 1)
    if std_w is None and dup_w is None:
        cov = torch.mean(  # (..., n_var, n_var)
            cent_obs @ cent_obs.transpose(-1, -2),
            axis=-3, keepdim=False  # (..., n_obs, n_var, n_var) -> (..., n_var, n_var)
        )
    else:
        cov = torch.sum(  # (..., n_var, n_var)
            weights_prod.unsqueeze(-1).unsqueeze(-1) * cent_obs @ cent_obs.transpose(-1, -2),
            axis=-3, keepdim=False  # (..., n_obs, n_var, n_var) -> (..., n_var, n_var)
        )
        cov /= torch.sum(dup_w, axis=-1, keepdim=False).unsqueeze(-1).unsqueeze(-1)
    return mean, cov


def _fit_n_clusters_one_step(
    obs: Tensor, std_w: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """One step of the EM algorithme.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    std_w : Tensor, optional
        The inv std weights of shape (..., nbr_observation).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    mean : Tensor
        The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
        This is an output parameter changed inplace.
    cov : Tensor
        The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
        This is an output parameter changed inplace.
    eta : Tensor
        The relative mass of each gaussian of shape (..., nbr_clusters).
        This is an output parameter changed inplace.

    Returns
    -------
    mean : Tensor
        A reference to the input parameter `mean`.
    cov : Tensor
        A reference to the input parameter `cov`.
    eta : Tensor
        A reference to the input parameter `eta`.

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    eps = torch.finfo(obs.dtype).eps

    # posterior probability that observation j belongs to cluster i
    post = _gauss(obs, mean, cov)  # (..., n_clu, n_obs)
    post += eps  # prevention again division by 0
    post *= eta.unsqueeze(-1)  # (..., n_clu, n_obs)
    post /= torch.sum(post, axis=-2, keepdim=True)  # (..., n_clu, n_obs)

    # update the relative mass of each gaussians (not normalized because it it used for cov)
    eta = torch.sum(dup_w.unsqueeze(-2) * post, axis=-1, keepdim=False, out=eta)  # (..., n_clu)

    # mean
    weighted_post = (std_w * dup_w).unsqueeze(-2) * post  # (..., n_clu, n_obs)
    weighted_post = weighted_post.unsqueeze(-1).unsqueeze(-1)  # (..., n_clu, n_obs, 1, 1)
    mean = torch.sum(
        weighted_post * obs.unsqueeze(-3).unsqueeze(-1),  # (..., n_clu, n_obs, n_var, 1)
        axis=-3, keepdim=False, out=mean  # (..., n_clus, n_var, 1)
    )
    mean /= torch.sum(weighted_post, axis=-3, keepdim=False)  # (..., n_clus, n_var, 1)

    # cov
    cent_obs = obs.unsqueeze(-1).unsqueeze(-4) - mean.unsqueeze(-3)  # (..., n_clu, n_obs, n_var, 1)
    cov = torch.sum(
        weighted_post * cent_obs @ cent_obs.transpose(-1, -2),  # (..., n_clu, n_obs, n_var, n_var)
        axis=-3, keepdim=False, out=cov  # (..., n_clu, n_var, n_var)
    )
    cov /= eta.unsqueeze(-1).unsqueeze(-1)

    # normalize eta
    eta /= torch.sum(dup_w.unsqueeze(-2), axis=-1, keepdim=False)  # (..., n_clu)

    return mean, cov, eta


def _fit_n_clusters_serveral_steps(
    obs: Tensor, std_w: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Same as `_fit_n_clusters_one_step` but iterates until it converges."""
    log_likelihood = _log_likelihood(obs, dup_w, mean, cov, eta)  # (...,)
    ongoing = torch.ones_like(log_likelihood, dtype=bool)  # mask for clusters converging

    for _ in range(1000):  # not while True for security
        new_mean, new_cov, new_eta = _fit_n_clusters_one_step(
            obs[ongoing], std_w[ongoing], dup_w[ongoing],
            mean[ongoing], cov[ongoing], eta[ongoing],
        )
        mean[ongoing], cov[ongoing], eta[ongoing] = new_mean, new_cov, new_eta
        new_log_likelihood = _log_likelihood(obs[ongoing], dup_w[ongoing], new_mean, new_cov, new_eta)
        converged = new_log_likelihood - log_likelihood[ongoing] <= 1e-3  # the mask of the converged clusters
        log_likelihood[ongoing] = new_log_likelihood
        if torch.all(converged):
            break
        # we want to do ongoing[ongoing][improvement] = False
        ongoing_ = ongoing[ongoing]
        ongoing_[converged] = False
        ongoing[ongoing.clone()] = ongoing_
        # print("ongoing", ongoing)  # to visualize the evolution of the convergence
    else:
        logging.warning("some gmm clusters failed to converge")

    return mean, cov, eta, log_likelihood



def em(
    obs: Tensor,
    std_w: typing.Optional[Tensor] = None,
    dup_w: typing.Optional[Tensor] = None,
    *,
    nbr_clusters: numbers.Integral = 1,
    nbr_tries: numbers.Integral = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    ** An implementation of the EM algorithm. **

    Notations
    ---------
    * \(
        \mathbf{\mu}_j =
        \begin{pmatrix}
            \mu_1  \\
            \vdots \\
            \mu_d  \\
        \end{pmatrix}_j
    \), the center of the gaussian \(j\), with \(d\) the space dimension or the number of variables.
    * \(
        \mathbf{\Sigma}_j =
        \begin{pmatrix}
            \sigma_{1,1} & \dots  & \sigma_{1,d} \\
            \vdots       & \ddots & \vdots       \\
            \sigma_{d,1} & \dots  & \sigma_{d,d} \\
        \end{pmatrix}_j
    \), the full symetric positive covariance matrix of the gaussian \(j\),
        with \(d\) the space dimension or the number of variables.
    * \(\eta_j\), the relative mass of the gaussian \(j\). We have \(\sum\limits_{j=1}^c \eta_j = 1\), with \(c\) the number of clusters.
    * \(
        \Gamma(\mathbf{x}) =
        \sum\limits_{j=1}^k \eta_j \mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}(\mathbf{x})
    \), the probability density at the point in space \(\mathbf{x}\).
    * \(k\), the number of Gaussians in the model.
    * \(\mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}\),
        a multidimensional Gaussian of type ``Gaussian``.

    Detailed algorithm for going from step \(s\) to step \(s+1\)
    ------------------------------------------------------------

    * \(
        p_{i,j} = \frac{
            \eta_j^{(s)}
            \mathcal{N}_{\mathbf{\mu}_j^{(s)}, \mathbf{\Sigma}_j^{(s)}}(\mathbf{x}_i)
        }{
            \sum\limits_{l=1}^k
            \eta_l^{(s)}
            \mathcal{N}_{\mathbf{\mu}_l^{(s)}, \mathbf{\Sigma}_l^{(s)}}(\mathbf{x}_i)
        }
    \) posterior probability that point \(i\) belongs to cluster \(j\)
    * \(
        \begin{cases}
            \eta_j^{(s+1)} =
                \frac{\sum\limits_{i=1}^n \alpha_i p_{i, j}}{\sum\limits_{i=1}^n \alpha_i},
                && \text{if weighting} \\
            \eta_j^{(s+1)} = \frac{1}{k} && \text{otherwise} \\
        \end{cases}
    \) the relative weight of each Gaussian
    * \(
        \mathbf{\mu_j}^{(s+1)} = \frac{
            \sum\limits_{i=1}^n \omega_i \alpha_i p_{i,j} \mathbf{x}_i
        }{
            \sum\limits_{i=1}^n \omega_i \alpha_i p_{i,j}
        }
    \) the mean of each gaussian
    * \(
        \mathbf{\Sigma}_j^{(s+1)} = \frac{
            \sum\limits_{i=1}^n
            \omega_i \alpha_i p_{i,j}
            (\mathbf{x}_i - \mathbf{\mu_j}^{(s+1)})
            (\mathbf{x}_i - \mathbf{\mu_j}^{(s+1)})^{\intercal}
        }{
            \sum\limits_{i=1}^n \alpha_i p_{i,j}
        }
    \) the cov of each gaussian

    Parameters
    ----------
    obs : arraylike
        \(\mathbf{x}\), The matrix of individuals.
    std_w : arraylike, optional
        \(\omega\), The vector of homogeneous weights
        has the inverse of the variance of each individual.
    dup_w : arraylike, optional
        \(\alpha\), The vector of weights which represents
        'the number of times the individual appears'.
    nbr_tries : int, default=1
        The number of times that the algorithm converges
        in order to have the best possible solution.

    Returns
    -------
    means : np.ndarray
        The matrix of the centers of the Gaussians.

        \(
            means =
            \begin{pmatrix}
                \mu_{1,1} & \dots  & \mu_{1,d} \\
                \vdots    & \ddots & \vdots    \\
                \mu_{k,1} & \dots  & \mu_{k,d} \\
            \end{pmatrix}
        \), ``shape=(n_components, space_dim)``
    covs : np.ndarray
        The tensor of covariance matrices.

        \(
            covs =
            \begin{bmatrix}
                \mathbf{\Sigma}_1 & \dots & \mathbf{\Sigma}_k
            \end{bmatrix}
        \), ``shape=(n_components, space_dim, space_dim)``
    """
    # verifications
    if not isinstance(obs, Tensor):
        obs = Tensor(obs)
    assert obs.ndim >= 2, obs.shape
    *batch, n_obs, n_var = obs.shape
    if std_w is not None:
        std_w = Tensor(std_w)
        assert std_w.ndim >= 1, std_w.shape
        *batch_w, n_obs_w = std_w.shape
        assert n_obs == n_obs_w, f"the number of observation doesn't match, {n_obs} vs {n_obs_w}"
        assert len(batch) == len(batch_w), f"batches are not broadcastables {batch} vs {batch_w}"
        try:
            torch.broadcast_shapes(batch, batch_w)
        except RuntimeError as err:
            raise AssertionError(f"batches are not broadcastables {batch} vs {batch_w}") from err
        assert std_w.dtype == obs.dtype, (std_w.dtype, obs.dtype)
        assert std_w.device == obs.device, (std_w.device, obs.device)
    if dup_w is not None:
        dup_w = Tensor(dup_w)
        assert dup_w.ndim >= 1, dup_w.shape
        *batch_w, n_obs_w = dup_w.shape
        assert n_obs == n_obs_w, f"the number of observation doesn't match, {n_obs} vs {n_obs_w}"
        assert len(batch) == len(batch_w), f"batches are not broadcastables {batch} vs {batch_w}"
        try:
            torch.broadcast_shapes(batch, batch_w)
        except RuntimeError as err:
            raise AssertionError(f"batches are not broadcastables {batch} vs {batch_w}") from err
        assert dup_w.dtype == obs.dtype, (dup_w.dtype, obs.dtype)
        assert dup_w.device == obs.device, (dup_w.device, obs.device)
    assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
    assert nbr_clusters > 0, nbr_clusters
    nbr_clusters = int(nbr_clusters)
    assert isinstance(nbr_tries, numbers.Integral), nbr_tries.__class__.__name__
    assert nbr_tries > 0, nbr_tries
    nbr_tries = int(nbr_tries)

    # general
    mean, cov = _fit_one_cluster(obs, std_w, dup_w)  # (..., n_var, 1) and (..., n_var, n_var)
    eta = torch.full((*batch, nbr_clusters), 1.0/nbr_clusters, dtype=obs.dtype, device=obs.device)
    if nbr_clusters != 1:
        # draw random points for initialization
        mean = (  # (n_tries, ..., n_clu, n_var, 1)
            mean.unsqueeze(-3).unsqueeze(0)
            + (
                torch.randn(
                    (nbr_tries, *batch, nbr_clusters, n_var),
                    dtype=obs.dtype, device=obs.device
                ) @ cov
            ).unsqueeze(-1)
        )
        cov = (  # (n_tries, ..., n_clu, n_var, n_var)
            cov
            .unsqueeze(-3).unsqueeze(0)
            .expand(nbr_tries, *batch, nbr_clusters, n_var, n_var)
            .clone()
        )
        eta = ( # (n_tries, ..., n_clu)
            eta.unsqueeze(0).expand(nbr_tries, *batch, nbr_clusters).clone()
        )
        # prepare data for several draw
        obs = obs.unsqueeze(0).expand(nbr_tries, *batch, n_obs, n_var)
        std_w = std_w.unsqueeze(0).expand(nbr_tries, *batch, n_obs)
        dup_w = dup_w.unsqueeze(0).expand(nbr_tries, *batch, n_obs)
        # run em on each cluster and each tries
        mean, cov, eta, criteria = (
            _fit_n_clusters_serveral_steps(obs, std_w, dup_w, mean, cov, eta)
        )
        # keep the best
        if nbr_clusters == 1:
            mean, cov, eta = mean.unsqueeze(0), cov.unsqueeze(0), eta.unsqueeze(0)
        else:
            criteria, best = torch.max(criteria, axis=0)
            mean = mean[best, range(mean.shape[1])]
            cov = cov[best, range(cov.shape[1])]
            eta = eta[best, range(eta.shape[1])]
    else:  # case one cluster
        mean = mean.unsqueeze(-3)  # (..., 1, n_var, 1)
        cov = cov.unsqueeze(-3)  # (..., 1, n_var, n_var)

    # finalization
    return mean, cov, eta  # (..., n_clu, n_var, 1), (..., n_clu, n_var, n_var), (..., n_clu)


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt
#     import time

#     # samples test
#     n_obs, n_var = 10000, 2
#     obs = torch.cat([  # (..., n_obs, n_var)
#         torch.randn((n_obs, n_var)) @ torch.tensor([[0.5, 0], [0, 0.5]]) + torch.tensor([[-1, -1]]),
#         # torch.randn((n_obs, n_var)) @ torch.tensor([[0.2, 0], [0, 0.7]]) + torch.tensor([[-1, 1]]),
#         torch.randn((n_obs, n_var)) @ torch.tensor([[0.4, -0.3], [-0.3, 0.4]]) + torch.tensor([[1, 1]]),
#         torch.randn((n_obs, n_var)) @ torch.tensor([[.1, 0], [0, .1]]) + torch.tensor([[1, -1]]),
#     ], axis=-2)
#     n_obs = obs.shape[-2]
#     std_w = torch.ones((n_obs,))  # (..., n_obs)
#     dup_w = torch.ones((n_obs,))  # (..., n_obs)
#     plt.scatter(obs[..., 0], obs[..., 1], label="observation")

#     obs, std_w, dup_w = obs.unsqueeze(0).expand(10, -1, -1).clone(), std_w.unsqueeze(0).expand(10, -1).clone(), dup_w.unsqueeze(0).expand(10, -1).clone()

#     estimated_mean, estimated_cov, estimated_eta = em(obs, std_w, dup_w, nbr_clusters=3, nbr_tries=2)
#     plt.scatter(estimated_mean[..., 0, 0], estimated_mean[..., 1, 0], label=f"final means")

#     plt.show()

