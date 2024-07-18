#!/usr/bin/env python3

"""Protection against the user."""

import functools
import warnings

import torch


def _assert_to_warn(checker):
    @functools.wrap(checker)
    def warn_checker(*args, **kwargs):
        try:
            checker(*args, **kwargs)
        except AssertionError as err:
            warnings.warn(err, UserWarning)
            return False
        return True
    return warn_checker


def check_gmm(gmm: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    r"""Ensure the provided parameters are corrects.

    Parameters
    ----------
    gmm : tuple of torch.Tensor
        * mean : torch.Tensor
            The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\)).
        * cov : torch.Tensor
            The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).
        * eta : torch.Tensor
            The relative mass \(\eta_j\) of shape (..., \(K\)).

    Raises
    ------
    AssertionError
        If the parameters are not correct.
    """
    # packaging check
    assert isinstance(gmm, tuple), f"gmm has to be a tuple, not a {gmm.__class__.__name__}"
    assert len(gmm) == 3, f"gmm has to contain 3 elements, not {len(gmm)}"

    # check dtype
    mean, cov, eta = gmm
    assert isinstance(mean, torch.Tensor), \
        f"mean has to be a torch tensor, not a {mean.__class__.__name__}"
    assert isinstance(cov, torch.Tensor), \
        f"cov has to be a torch tensor, not a {cov.__class__.__name__}"
    assert isinstance(eta, torch.Tensor), \
        f"eta has to be a torch tensor, not a {eta.__class__.__name__}"

    # check shape
    assert mean.ndim >= 2, f"mean has to be of shape (..., K, D, 1), not {mean.shape}"
    assert cov.ndim >= 3, f"cov has to be of shape (..., N, D, D), not {cov.shape}"
    assert eta.ndim >= 1, f"eta has to be a shape (..., K), np {eta.shape}"
    *mean_batch, mean_K, mean_D = mean.shape
    *cov_batch, cov_K, cov_D1, cov_D2 = cov.shape
    *eta_batch, eta_K = eta.shape
    assert mean_batch == cov_batch == eta_batch, \
        f"batch dimension dosent match {mean_batch} vs {cov_batch} vs {eta_batch}"
    assert mean_D == cov_D1 == cov_D2, \
        f"space dimension inconsistant {mean_D} vs {cov_D1} vs {cov_D2}"
    assert mean_K == cov_K == eta_K, \
        f"number of gaussians inconsistant {mean_K} vs {cov_K} vs {eta_K}"


def check_infit(obs: torch.Tensor, weights: None | torch.Tensor) -> None:
    r"""Ensure the provided parameters are corrects.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    weights : torch.Tensor, optional
        The duplication weights of shape (..., \(N\)).

    Raises
    ------
    AssertionError
        If the parameters are not correct.
    """
    # check dtype
    assert isinstance(obs, torch.Tensor), \
        f"obs has to be a torch tensor, not a {obs.__class__.__name__}"
    assert weights is None or isinstance(weights, torch.Tensor), \
        f"weights has to be a torch tensor, not a {weights.__class__.__name__}"

    # check shape
    assert obs.ndim >= 2, f"obs has to be of shape (..., N, D), not {obs.shape}"
    *obs_batch, obs_N, _ = obs.shape
    if weights is not None:
        assert weights.ndim >= 1, \
            f"dup_w and std_w has to be of shape (..., N), not {weights.shape}"
        *w_batch, w_N = weights.shape
        assert obs_batch == w_batch, f"batch dimension dosent match {obs_batch} vs {w_batch}"
        assert obs_N == w_N, f"number of observations inconsistant {obs_N} vs {w_N}"


def check_ingauss(obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> None:
    r"""Ensure the provided parameters are corrects.

    Parameters
    ----------
    obs : torch.Tensor
        The observations \(\mathbf{x}_i\) of shape (..., \(N\), \(D\)).
    mean : torch.Tensor
        The column mean vector \(\mathbf{\mu}_j\) of shape (..., \(K\), \(D\)).
    cov : torch.Tensor
        The covariance matrix \(\mathbf{\Sigma}\) of shape (..., \(K\), \(D\), \(D\)).

    Raises
    ------
    AssertionError
        If the parameters are not correct.
    """
    # check dtype
    assert isinstance(obs, torch.Tensor), \
        f"obs has to be a torch tensor, not a {obs.__class__.__name__}"
    assert isinstance(mean, torch.Tensor), \
        f"mean has to be a torch tensor, not a {mean.__class__.__name__}"
    assert isinstance(cov, torch.Tensor), \
        f"cov has to be a torch tensor, not a {cov.__class__.__name__}"

    # check shape
    assert obs.ndim >= 2, f"obs has to be of shape (..., N, D), not {obs.shape}"
    assert mean.ndim >= 2, f"mean has to be of shape (..., K, D, 1), not {mean.shape}"
    assert cov.ndim >= 3, f"cov has to be of shape (..., N, D, D), not {cov.shape}"
    *obs_batch, _, obs_D = obs.shape
    *mean_batch, mean_K, mean_D = mean.shape
    *cov_batch, cov_K, cov_D1, cov_D2 = cov.shape
    assert obs_batch == mean_batch == cov_batch, \
        f"batch dimension dosent match {obs_batch} vs {mean_batch} vs {cov_batch}"
    assert obs_D == mean_D == cov_D1 == cov_D2, \
        f"space dimension inconsistant {obs_D} vs {mean_D} vs {cov_D1} vs {cov_D2}"
    assert mean_K == cov_K, f"number of gaussians inconsistant {mean_K} vs {cov_K}"
