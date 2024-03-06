#!/usr/bin/env python3

"""Gaussian mixture model fited by the expequancy maximisation."""


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
    ind_prob = torch.log(ind_prob, out=ind_prob)
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
    prob = torch.exp(prob, out=prob)
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
    weights_prod = std_w * dup_w  # (..., n_obs)
    mean = torch.sum((weights_prod.unsqueeze(-1) * obs), axis=-2, keepdim=True)  # (..., 1, n_var)
    mean /= torch.sum(weights_prod.unsqueeze(-1), axis=-2, keepdim=True)  # (..., 1, n_var)
    cent_obs = (obs - mean).unsqueeze(-1)  # (..., n_obs, n_var, 1)
    mean = mean.squeeze(-2).unsqueeze(-1)  # (..., n_var, 1)
    cov = torch.sum(  # (..., n_var, n_var)
        weights_prod.unsqueeze(-1).unsqueeze(-1) * cent_obs @ cent_obs.transpose(-1, -2),
        axis=-3, keepdim=False  # (..., n_obs, n_var, n_var) -> (..., n_var, n_var)
    )
    cov /= torch.sum(dup_w, axis=-1, keepdim=False).unsqueeze(-1).unsqueeze(-1)
    return mean, cov


def _fit_n_clusters_one_step(
    obs: Tensor, std_w: Tensor, dup_w: Tensor, mean: Tensor, cov: Tensor, eta: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    post : Tensor
        Posterior probability that observation j belongs to cluster i.
        Shape (..., nbr_clusters, nbr_observation).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    # posterior probability that observation j belongs to cluster i
    post = _gauss(obs, mean, cov)  # (..., n_clu, n_obs)
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

    return mean, cov, eta, post


def em(
    obs: Tensor, *,
    std_w: typing.Optional[Tensor] = None,
    dup_w: typing.Optional[Tensor] = None,
    nbr_clusters: numbers.Integral = 1,
    nbr_tries: numbers.Integral = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    ** An implementation of the EM algorithm. **

    Notations
    ---------
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
    weights : np.ndarray
        The vectors of relative weights associated with each Gaussian.

        \(
            weights =
            \begin{pmatrix}
                \eta_1 & \dots & \eta_k
            \end{pmatrix}
        \) with \(
            \begin{cases}
                \sum\limits_{j=1}^k \eta_j = 1 \\
                \eta_j > 0, \forall j \in [\!\![ 1, k ]\!\!] \\
            \end{cases}
        \), ``shape=(n_components,)``
    """
    # verifications
    if not isinstance(obs, Tensor):
        obs = Tensor(obs)
    assert obs.ndim >= 2, obs.shape
    *batch_obs, n_obs, n_var = obs.shape
    if std_w is not None:
        std_w = Tensor(std_w)
        assert std_w.ndim >= 1, std_w.shape
        *batch_w, n_obs_w = std_w.shape
        assert n_obs == n_obs_w, f"the number of observation doesn't match, {n_obs} vs {n_obs_w}"
        assert len(batch_obs) == len(batch_w), \
            f"batches are not broadcastables {batch_obs} vs {batch_w}"
        try:
            torch.broadcast_shapes(batch_obs, batch_w)
        except RuntimeError as err:
            raise AssertionError(
                f"batches are not broadcastables {batch_obs} vs {batch_w}"
            ) from err
        assert std_w.dtype == obs.dtype, (std_w.dtype, obs.dtype)
        assert std_w.device == obs.device, (std_w.device, obs.device)
    if dup_w is not None:
        dup_w = Tensor(dup_w)
        assert dup_w.ndim >= 1, dup_w.shape
        *batch_w, n_obs_w = dup_w.shape
        assert n_obs == n_obs_w, f"the number of observation doesn't match, {n_obs} vs {n_obs_w}"
        assert len(batch_obs) == len(batch_w), \
            f"batches are not broadcastables {batch_obs} vs {batch_w}"
        try:
            torch.broadcast_shapes(batch_obs, batch_w)
        except RuntimeError as err:
            raise AssertionError(
                f"batches are not broadcastables {batch_obs} vs {batch_w}"
            ) from err
        assert dup_w.dtype == obs.dtype, (dup_w.dtype, obs.dtype)
        assert dup_w.device == obs.device, (dup_w.device, obs.device)
    assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
    assert nbr_clusters > 0, nbr_clusters
    nbr_clusters = int(nbr_clusters)
    assert isinstance(nbr_tries, numbers.Integral), nbr_tries.__class__.__name__
    assert nbr_tries > 0, nbr_tries
    nbr_tries = int(nbr_tries)

    # initialization
    mean, cov = _fit_one_cluster(obs, std_w, dup_w)  # (..., n_var, 1) and (..., n_var, n_var)

    raise NotImplementedError


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    # samples test
    n_obs, n_var = 10000, 2
    obs = torch.cat([  # (..., n_obs, n_var)
        torch.randn((n_obs, n_var)) @ torch.tensor([[0.5, 0], [0, 0.5]]) + torch.tensor([[-1, -1]]),
        # torch.randn((n_obs, n_var)) @ torch.tensor([[0.2, 0], [0, 0.7]]) + torch.tensor([[-1, 1]]),
        torch.randn((n_obs, n_var)) @ torch.tensor([[0.4, -0.3], [-0.3, 0.4]]) + torch.tensor([[1, 1]]),
        torch.randn((n_obs, n_var)) @ torch.tensor([[.1, 0], [0, .1]]) + torch.tensor([[1, -1]]),
    ], axis=-2)
    n_obs = obs.shape[-2]
    std_w = torch.ones((n_obs,))  # (..., n_obs)
    dup_w = torch.ones((n_obs,))  # (..., n_obs)
    plt.scatter(obs[..., 0], obs[..., 1], label="observation")

    # initialization
    mean_init, cov_init = _fit_one_cluster(obs, std_w, dup_w)  # (..., n_var, 1) and (..., n_var, n_var)

    for n_clu in range(1, 7):

        eta = torch.ones((*cov_init.shape[:-2], n_clu)) / n_clu
        # plt.scatter(mean_init[..., 0, 0], mean_init[..., 1, 0], label="mean init ref")

        means = mean_init.unsqueeze(-3) + (torch.randn((*mean_init.shape[:-2], n_clu, n_var)) @ cov_init).unsqueeze(-1)
        covs = cov_init.expand(*cov_init.shape[:-2], n_clu, -1, -1).clone()
        etas = eta.clone()
        # plt.scatter(means[..., 0, 0], means[..., 1, 0], label=f"means init {n_clu} clusters")

        # feet n clusters
        means_hash = {hash(pickle.dumps(means))}
        for i in range(1000):
            means, covs, etas, _ = _fit_n_clusters_one_step(obs, std_w, dup_w, means, covs, etas)
            if torch.any(torch.isnan(means)):
                print("hahahahahaha")
            if (new_hash := hash(pickle.dumps(means))) in means_hash:
                plt.scatter(means[..., 0, 0], means[..., 1, 0], label=f"final means, {n_clu} clusters and {i} iterations")
                break
            means_hash.add(new_hash)

        # evaluation of quality
        bic = _bic(obs, dup_w, means, covs, etas)
        aic = _aic(obs, dup_w, means, covs, etas)
        print(f"for {n_clu} clusters, bic={bic:.2f}, aic={aic:.2f}")

    plt.legend()
    plt.show()
