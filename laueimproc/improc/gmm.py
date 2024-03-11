#!/usr/bin/env python3

r"""Gaussian mixture model fited by the expequancy maximisation.


Scalar Terminology
------------------
* \(D\): The space dimension, same as the number of variables. Index \(d \in [\![1;D]\!]\).
* \(K\): The number of gaussians. Index \(j \in [\![1;K]\!]\).
* \(N\): The number of observations. Index \(i \in [\![1;N]\!]\).

Tensor Terminology
------------------
* \(
    \mathbf{\mu}_j =
    \begin{pmatrix}
        \mu_1  \\
        \vdots \\
        \mu_D  \\
    \end{pmatrix}_j
\): The center of the gaussian \(j\).
* \(
    \mathbf{\Sigma}_j =
    \begin{pmatrix}
        \sigma_{1,1} & \dots  & \sigma_{1,D} \\
        \vdots       & \ddots & \vdots       \\
        \sigma_{D,1} & \dots  & \sigma_{d,D} \\
    \end{pmatrix}_j
\): The full symetric positive covariance matrix of the gaussian \(j\).
* \(\eta_j\), the relative mass of the gaussian \(j\). We have \(\sum\limits_{j=1}^K \eta_j = 1\).
* \(\alpha_i\): The weights has the relative number of time the individual has been drawn.
* \(\omega_i\): The weights has the inverse of the relative covariance of each individual.
* \(
    \mathbf{x}_i =
    \begin{pmatrix}
        x_1    \\
        \vdots \\
        x_D    \\
    \end{pmatrix}_j
\): The observation \(i\).

Calculus Terminology
--------------------
* \(
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
\): The multidimensional gaussian probability density.
* \(
    \Gamma(\mathbf{x}_i) =
    \sum\limits_{j=1}^K \eta_j \mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}(\mathbf{x}_i)
\): The total probability density of the observation \(\mathbf{x}_i\).
"""

import collections
import logging
import math
import numbers
import typing

import torch

from laueimproc.classes.tensor import Tensor


GMM = collections.namedtuple("GMM", (  # It is just a compact way to pack gmm params.
    "mean",  # The column mean vector of shape (..., nbr_clusters, nbr_variables, 1).
    "cov",  # The covariance matrix of shape (..., nbr_clusters, nbr_variables, nbr_variables).
    "eta",  # The relative mass of each gaussian of shape (..., nbr_clusters).
))
__pdoc__ = {"GMM": False}


def _aic(obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM, *, log_likelihood=None) -> Tensor:
    """Akaike Information Criterion.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    std_w : Tensor, optional
        The inverse var weights of shape (..., nbr_observation).
    gmm : GMM
        The gaussians mixture params (mean, cov and eta).

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
    ) * gmm.eta.shape[-1]  # nbr of gaussians
    aic = (
        _log_likelihood(obs, dup_w, std_w, gmm)
        if log_likelihood is None else
        log_likelihood.clone()
    )
    aic *= -2.0
    aic += 2 * float(free_parameters)
    return aic


def _bic(obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM, *, log_likelihood=None) -> Tensor:
    """Compute the Bayesian Information Criterion.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    gmm : GMM
        The gaussians mixture params (mean, cov and eta).

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
    ) * gmm.eta.shape[-1]  # nbr of gaussians
    if dup_w is None:
        log_n_obs = math.log(obs.shape[-2])
    else:
        log_n_obs = torch.sum(dup_w, axis=-1, keepdim=False)
        log_n_obs = torch.log(log_n_obs, out=log_n_obs)
    bic = (
        _log_likelihood(obs, dup_w, std_w, gmm)
        if log_likelihood is None else
        log_likelihood.clone()
    )
    bic *= -2.0
    bic += log_n_obs * float(free_parameters)
    return bic


def _check(
    obs: typing.Container,
    dup_w: typing.Optional[typing.Container] = None,
    std_w: typing.Optional[typing.Container] = None,
) -> tuple[Tensor, typing.Union[None, Tensor], typing.Union[None, Tensor]]:
    """Check and convert into Tensor."""
    if not isinstance(obs, Tensor):
        obs = Tensor(obs)
    assert obs.ndim >= 2, obs.shape
    *batch, n_obs, _ = obs.shape

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

    return obs, dup_w, std_w


def _log_likelihood(obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM) -> Tensor:
    """Compute the log likelihood.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    gmm : GMM
        The gaussians mixture params (mean, cov and eta).

    Returns
    -------
    log_likelihood : Tensor
        A real positive scalar of shape (...,).

    Notes
    -----
    * No verifications are performed for performance reason.
    """
    if std_w is not None:
        raise NotImplementedError("std_w is not implemented in the likelihood computation")
    mean, cov, eta = gmm
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
    if prob.requires_grad:
        prob = prob * norm
    else:
        prob *= norm
    return prob


def _fit_one_cluster(obs: Tensor, dup_w: Tensor, std_w: Tensor) -> tuple[Tensor, Tensor]:
    """Fit one gaussian.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    std_w : Tensor, optional
        The inverse var weights of shape (..., nbr_observation).

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
        (1.0 if dup_w is None else dup_w)
        * (1.0 if std_w is None else std_w)
    )
    if dup_w is None and std_w is None:
        mean = torch.mean(obs, axis=-2, keepdim=True)  # (..., 1, n_var)
    else:
        mean = torch.sum((weights_prod.unsqueeze(-1) * obs), axis=-2, keepdim=True)
        mean /= torch.sum(weights_prod.unsqueeze(-1), axis=-2, keepdim=True)  # (..., 1, n_var)
    cent_obs = (obs - mean).unsqueeze(-1)  # (..., n_obs, n_var, 1)
    mean = mean.squeeze(-2).unsqueeze(-1)  # (..., n_var, 1)
    if dup_w is None and std_w is None:
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


def _fit_n_clusters_one_step(obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM) -> GMM:
    """One step of the EM algorithme.

    Parameters
    ----------
    obs : Tensor
        The observations of shape (..., nbr_observation, nbr_variables).
    dup_w : Tensor, optional
        The duplication weights of shape (..., nbr_observation).
    std_w : Tensor, optional
        The inverse var weights of shape (..., nbr_observation).
    gmm : GMM
        The gaussians mixture params (mean, cov and eta).

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
    mean, cov, eta = gmm
    eps = torch.finfo(obs.dtype).eps

    # posterior probability that observation j belongs to cluster i
    post = _gauss(obs, mean, cov)  # (..., n_clu, n_obs)
    post += eps  # prevention again division by 0
    post *= eta.unsqueeze(-1)  # (..., n_clu, n_obs)
    post /= torch.sum(post, axis=-2, keepdim=True)  # (..., n_clu, n_obs)

    # update the relative mass of each gaussians (not normalized because it it used for cov)
    eta = torch.sum(dup_w.unsqueeze(-2) * post, axis=-1, keepdim=False, out=eta)  # (..., n_clu)

    # mean
    weighted_post = (dup_w * std_w).unsqueeze(-2) * post  # (..., n_clu, n_obs)
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

    return GMM(mean, cov, eta)


def _fit_n_clusters_serveral_steps(
    obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM
) -> tuple[GMM, Tensor]:
    """Same as `_fit_n_clusters_one_step` but iterates until it converges."""
    mean, cov, eta = gmm
    log_likelihood = _log_likelihood(obs, dup_w, std_w, gmm)  # (...,)
    ongoing = torch.ones_like(log_likelihood, dtype=bool)  # mask for clusters converging

    for _ in range(1000):  # not while True for security
        new_gmm = _fit_n_clusters_one_step(
            obs[ongoing], dup_w[ongoing], std_w[ongoing],
            GMM(mean[ongoing], cov[ongoing], eta[ongoing]),
        )
        mean[ongoing], cov[ongoing], eta[ongoing] = new_gmm
        new_log_likelihood = _log_likelihood(obs[ongoing], dup_w[ongoing], std_w[ongoing], new_gmm)
        converged = (   # the mask of the converged clusters
            new_log_likelihood - log_likelihood[ongoing] <= 1e-3
        )
        log_likelihood[ongoing] = new_log_likelihood
        if torch.all(converged):
            break
        # we want to do equivalent of ongoing[ongoing][improvement] = False
        ongoing_ = ongoing[ongoing]
        ongoing_[converged] = False
        ongoing[ongoing.clone()] = ongoing_
    else:
        logging.warning("some gmm clusters failed to converge after 1000 iterations")

    return GMM(mean, cov, eta), log_likelihood


def _fit_n_clusters_serveral_steps_serveral_tries(
    nbr_tries: int, obs: Tensor, dup_w: Tensor, std_w: Tensor, gmm: GMM
) -> tuple[GMM, Tensor]:
    """Same as `_fit_n_clusters_serveral_steps` but draw several tries."""
    *batch, n_obs, n_var = obs.shape
    nbr_clusters = gmm.eta.shape[-1]

    # draw random initial conditions
    mean = (  # (n_tries, ..., n_clu, n_var, 1)
        gmm.mean.unsqueeze(-3).unsqueeze(0)
        + (
            torch.randn(
                (nbr_tries, *batch, nbr_clusters, n_var),
                dtype=obs.dtype, device=obs.device
            ) @ gmm.cov
        ).unsqueeze(-1)
    )
    cov = (  # (n_tries, ..., n_clu, n_var, n_var)
        gmm.cov
        .unsqueeze(-3).unsqueeze(0)
        .expand(nbr_tries, *batch, nbr_clusters, n_var, n_var)
        .clone()
    )
    eta = ( # (n_tries, ..., n_clu)
        gmm.eta.unsqueeze(0).expand(nbr_tries, *batch, nbr_clusters).clone()
    )

    # prepare data for several draw
    obs = obs.unsqueeze(0).expand(nbr_tries, *batch, n_obs, n_var)
    dup_w = dup_w.unsqueeze(0).expand(nbr_tries, *batch, n_obs)
    std_w = std_w.unsqueeze(0).expand(nbr_tries, *batch, n_obs)

    # run em on each cluster and each tries
    (mean, cov, eta), log_likelihood = (
        _fit_n_clusters_serveral_steps(obs, dup_w, std_w, GMM(mean, cov, eta))
    )

    # keep the best
    if nbr_clusters == 1:
        mean, cov, eta = mean.unsqueeze(0), cov.unsqueeze(0), eta.unsqueeze(0)
    else:
        log_likelihood, best = torch.max(log_likelihood, axis=0)
        mean = mean[best, range(mean.shape[1])]
        cov = cov[best, range(cov.shape[1])]
        eta = eta[best, range(eta.shape[1])]
    return GMM(mean, cov, eta), log_likelihood

def _mse(obs: Tensor, dup_w: Tensor, gmm: GMM) -> Tensor:
    """Return the mean square error."""
    mean, cov, eta = gmm
    prob = _gauss(obs, mean, cov)  # (..., n_clu, n_pxl)
    prob *= eta.unsqueeze(-1)  # relative proba
    mass = torch.sum(dup_w, axis=-1, keepdim=True)  # (..., 1)
    surface = torch.sum(prob, axis=-2)  # (..., n_pxl)
    surface *= mass
    mse = torch.mean((surface - dup_w)**2, axis=-1)  # (...,)
    return mse

def em(
    obs: Tensor,
    dup_w: typing.Optional[Tensor] = None,
    std_w: typing.Optional[Tensor] = None,
    *,
    nbr_clusters: numbers.Integral = 1,
    nbr_tries: numbers.Integral = 2,
    **kwargs,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    r"""A weighted implementation of the EM algorithm.


    Detailed algorithm for going from step \(s\) to step \(s+1\)
    ------------------------------------------------------------

    * \(
        p_{i,j} = \frac{
            \eta_j^{(s)}
            \mathcal{N}_{\mathbf{\mu}_j^{(s)}, \mathbf{\Sigma}_j^{(s)}}\left(\mathbf{x}_i\right)
        }{
            \sum\limits_{k=1}^K
            \eta_k^{(s)}
            \mathcal{N}_{\mathbf{\mu}_k^{(s)}, \mathbf{\Sigma}_k^{(s)}}\left(\mathbf{x}_i\right)
        }
    \) Posterior probability that observation \(i\) belongs to cluster \(j\).
    * \(
        \eta_j^{(s+1)} = \frac{\sum\limits_{i=1}^N \alpha_i p_{i,j}}{\sum\limits_{i=1}^N \alpha_i}
    \) The relative weight of each gaussian.
    * \(
        \mathbf{\mu}_j^{(s+1)} = \frac{
            \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j} \mathbf{x}_i
        }{
            \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j}
        }
    \) The mean of each gaussian.
    * \(
        \mathbf{\Sigma}_j^{(s+1)} = \frac{
            \sum\limits_{i=1}^N
            \omega_i \alpha_i p_{i,j}
            \left(\mathbf{x}_i - \mathbf{\mu}_j^{(s+1)}\right)
            \left(\mathbf{x}_i - \mathbf{\mu}_j^{(s+1)}\right)^{\intercal}
        }{
            \sum\limits_{i=1}^N \alpha_i p_{i,j}
        }
    \) The cov of each gaussian.

    This sted is iterated as long as the log likelihood increases.

    Parameters
    ----------
    obs : arraylike
        The observations \(\mathbf{x}\). Shape (..., \(N\), \(D\)).
    dup_w : arraylike, optional
        The weights \(\alpha\) has the relative number of time the individual has been drawn.
        Shape (..., \(N\)).
    std_w : arraylike, optional
        The weights \(\omega\) has the inverse of the standard deviation of each individual.
        Shape (..., \(N\)).
    nbr_clusters : int, default=1
        The number \(K\) of gaussians.
    nbr_tries : int, default=2
        The number of times that the algorithm converges
        in order to have the best possible solution.
        It is ignored in the case `nbr_clusters` = 1.
    aic, bic, log_likelihood, mean_std, mse : boolean, default=False
        If set to True, the metric is happend into `infodict`. Shape (...,).

        * aic: Akaike Information Criterion. \(aic = 2p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the log likelihood.
        * bic: Bayesian Information Criterion. \(bic = \log(N)p-2\log(L_{\alpha,\omega})\),
        \(p\) is the number of free parameters and \(L_{\alpha,\omega}\) the log likelihood.
        * log_likelihood: \(
            L_{\alpha,\omega} = \log\left(
                \prod\limits_{i=1}^N \sum\limits_{j=1}^K
                \eta_j \left(
                    \mathcal{N}_{(\mathbf{\mu}_j,\frac{1}{\omega_i}\mathbf{\Sigma}_j)}(\mathbf{x}_i)
                \right)^{\alpha_i}
            \right)
        \)
        * mean_std: The std of the mean estimator.
        Let \(\widehat{\mathbf{\mu}}\) be the estimator of the mean.

            * \(\begin{cases}
                std(\widehat{\mathbf{\mu}_j}) = \sqrt{var(\widehat{\mathbf{\mu}})} \\
                var(\widehat{\mathbf{\mu}_j}) = var\left( \frac{
                        \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j} \mathbf{x}_i
                    }{
                        \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j}
                    } \right) = \frac{
                        \sum\limits_{i=1}^N (\alpha_i \omega_i p_{i,j})^2 var(\mathbf{x}_i)
                    }{
                        \left( \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j} \right)^2
                    } \\
                var(\mathbf{x}_i) = max(eigen( \frac{1}{\omega_i}\mathbf{\Sigma}_j )) \\
            \end{cases}\)
            * \(
                std(\widehat{\mathbf{\mu}_j}) = \frac{
                    \sqrt{
                        max(eigen( \mathbf{\Sigma}_j ))
                        \sum\limits_{i=1}^N (\alpha_i p_{i,j})^2 \omega_i
                    }
                }{
                    \left( \sum\limits_{i=1}^N \alpha_i \omega_i p_{i,j} \right)
                }
            \)

        * mse: Mean Square Error. \(
            mse = \frac
                {
                    \sum\limits_{i=1}^N
                    \left(
                        \left(\sum\limits_{i=1}^N\alpha_i\right)\Gamma(\mathbf{X}_i)-\alpha_i
                    \right)^2
                }
                {\sum\limits_{i=1}^N\alpha_i}
        \)

    Returns
    -------
    mean : Tensor
        The vectors \(\mathbf{\mu}\). Shape (..., \(K\), \(D\), 1).
    cov : Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (..., \(K\), \(D\), \(D\)).
    eta : Tensor
        The relative mass \(\eta\). Shape (.., \(K\)).
    infodict : dict[str]
        A dictionary of optional outputs.
    """
    metrics = {"aic", "bic", "log_likelihood", "mean_std", "mse"}

    # verifications
    if kwargs.get("_check", True):
        obs, dup_w, std_w = _check(obs, dup_w, std_w)
        assert isinstance(nbr_clusters, numbers.Integral), nbr_clusters.__class__.__name__
        assert nbr_clusters > 0, nbr_clusters
        nbr_clusters = int(nbr_clusters)
        assert isinstance(nbr_tries, numbers.Integral), nbr_tries.__class__.__name__
        assert nbr_tries > 0, nbr_tries
        nbr_tries = int(nbr_tries)
        assert all(isinstance(kwargs.get(k, False), bool) for k in metrics), kwargs
        assert set(kwargs).issubset(metrics|{"_check", "mean_std"}), \
            f"unrecognise parameter {set(kwargs)-metrics}"

    # main gmm
    *batch, _, _ = obs.shape
    mean, cov = _fit_one_cluster(obs, dup_w, std_w)  # (..., n_var, 1) and (..., n_var, n_var)
    eta = torch.full((*batch, nbr_clusters), 1.0/nbr_clusters, dtype=obs.dtype, device=obs.device)
    log_likelihood = None
    if nbr_clusters != 1:
        (mean, cov, eta), log_likelihood = _fit_n_clusters_serveral_steps_serveral_tries(
            nbr_tries, obs, dup_w, std_w, (mean, cov, eta)
        )
    else:  # case one cluster
        mean = mean.unsqueeze(-3)  # (..., 1, n_var, 1)
        cov = cov.unsqueeze(-3)  # (..., 1, n_var, n_var)
        if metrics - kwargs.keys():
            log_likelihood = _log_likelihood(obs, dup_w, std_w, GMM(mean, cov, eta))

    # finalization
    infodict = {}
    if kwargs.get("aic", False):
        infodict["aic"] = _aic(
            obs, dup_w, std_w, GMM(mean, cov, eta), log_likelihood=log_likelihood
        )
    if kwargs.get("bic", False):
        infodict["bic"] = _bic(
            obs, dup_w, std_w, GMM(mean, cov, eta), log_likelihood=log_likelihood
        )
    if kwargs.get("log_likelihood", False):
        infodict["log_likelihood"] = log_likelihood
    if kwargs.get("mse", False):
        infodict["mse"] = _mse(obs, dup_w, GMM(mean, cov, eta))
    if kwargs.get("mean_std", False):
        logging.warning("wrong and stupid mean std estimation!!!!")
        assert nbr_clusters == 1, "notimplemented"
        eig = torch.linalg.eigvalsh(cov)  # (..., n_clu, n_var)
        eig = torch.max(eig, axis=-1).values  # (..., n_clu)
        eig = eig.squeeze(-1)  # (...,)
        mass = torch.sum(dup_w, axis=-1)  # (...,)
        infodict["mean_std"] = torch.sqrt(eig / mass)
    # if kwargs.get("tol", False):
    #     # def local_mse(mean):  # (..., n_clu, n_var, 1)
    #     #     return torch.sum(_mse(obs, dup_w, mean, cov, eta))
    #     # infodict["tol"] = torch.autograd.functional.hessian(local_mse, mean)
    #     mse = _mse(obs, dup_w, mean, cov, eta)
    #     d1mse = torch.autograd.grad(torch.sum(mse), mean, create_graph=True)[0]
    #     d2mse = torch.autograd.grad(torch.sum(torch.abs(d1mse)), mean)[0]
    #     infodict["tol"] = d2mse / mse
    #     # torch.sum(infodict["mse"]).backward()
    #     # infodict["tol"] = mean.grad / infodict["mse"]

    return collections.namedtuple(
        "GMMSolution", ("mean", "cov", "eta", "infodict")
    )(mean, cov, eta, infodict)
