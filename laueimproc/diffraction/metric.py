#!/usr/bin/env python3

"""Implement some loss functions."""

import functools
import logging
import math
import multiprocessing.pool
import numbers
import threading

import numpy as np
import torch

try:
    from laueimproc.diffraction import c_metric
except ImportError:
    logging.warning(
        "failed to import laueimproc.diffraction.c_metric, a slow python version is used instead"
    )
    c_metric = None
from .projection import detector_to_ray


def _cached_ray_to_table(func):
    """Decorate ray_to_table to cache the result."""
    @functools.wraps(func)
    def decorated(
        ray: torch.Tensor, res: numbers.Real
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """Invoque ``c_metric.ray_to_table`` and keep result in cache.

        Parameters
        ----------
        ray : torch.Tensor
            The rays of shape (n, 3).
        res : float
            The maximal sample step in spherical hash table projection.

        Returns
        -------
        table : np.ndarray[int]
            The hash table of shape (a, b, 2).
            The first layer ``table[:, :, 0]`` corresponds to the start inex of ``indicies``.
            The second layer ``table[:, :, 1]`` corresponds to the number of items.
        indices : np.ndarray[int]
            Matching between the table and the ray index.
            We have ``one_ray = ray[indices[table[c, d, 0] + i]]``,
            with 0 <= c < a, 0 <= d < b and 0 <= i < table[c, d, 1]
        limits : tuple[int, int, int, int]
            Internal padding consideration.
        """
        assert isinstance(ray, torch.Tensor), ray.__class__.__name__
        assert ray.ndim == 2 and ray.shape[1] == 3, ray.shape
        signature = (ray.data_ptr(), ray.shape[0], float(res))
        if signature not in cache:
            if len(cache) > 64:  # to avoid memory leaking
                cache.clear()
            cache[signature] = c_metric.ray_to_table(ray.numpy(force=True), res)
        return cache[signature]
    cache = {}
    return decorated


ray_to_table = None if c_metric is None else _cached_ray_to_table(c_metric.ray_to_table)


def raydotdist(
    ray_point_1: torch.Tensor, ray_point_2: torch.Tensor, poni: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Compute the scalar product matrix of the rays pairwise.

    Parameters
    ----------
    ray_point_1 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (*n, r1, 2).
        Could be directly the unitary ray vector uf or uq of shape (*n, r1, 3).
    ray_point_2 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (*n', r2, 2)).
        Could be directly the unitary ray vector uf or uq of shape (*n', r2, 3).
    poni : torch.Tensor, optional
        Only use if the ray are projected points.
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (*p, 6).

    Returns
    -------
    dist : torch.Tensor
        The distance matrix \(\cos(\phi)\) of shape (*broadcast(n, n'), *p, r1, r2).
    """
    assert isinstance(ray_point_1, torch.Tensor), ray_point_1.__class__.__name__
    assert isinstance(ray_point_2, torch.Tensor), ray_point_2.__class__.__name__
    assert ray_point_1.ndim >= 2 and ray_point_1.shape[-1] in {2, 3}, ray_point_1.shape
    assert ray_point_2.ndim >= 2 and ray_point_2.shape[-1] in {2, 3}, ray_point_2.shape

    if ray_point_1.shape[-1] == 2:
        assert poni is not None
        ray_1 = detector_to_ray(ray_point_1, poni)  # (*broadcast(p, p'), *r1, 3)
    else:
        ray_1 = ray_point_1
    if ray_point_2.shape[-1] == 2:
        assert poni is not None
        ray_2 = detector_to_ray(ray_point_2, poni)  # (*broadcast(p, p'), *r2, 3)
    else:
        ray_2 = ray_point_2

    ray_1 = ray_1[..., :, None, :]
    ray_2 = ray_2[..., None, :, :]
    dist = torch.sum(ray_1 * ray_2, dim=-1)  # <ray_1, ray_2> = cos(angle(ray_1, ray_2)) = cos(phi)
    return dist


def compute_matching_rate(
    exp_uq: torch.Tensor, theo_uq: torch.Tensor, phi_max: numbers.Real, *, _no_c: bool = False
) -> torch.Tensor:
    """Compute the matching rate.

    It is the number of ray in ``theo_uq``, close engouth to at least one ray of ``exp_uq``.

    Parameters
    ----------
    exp_uq : torch.Tensor
        The unitary experimental uq vector of shape (*n, e, 3).
    theo_uq : torch.Tensor
        The unitary simulated theorical uq vector of shape (*n', t, 3).
    phi_max : float
        The maximum positive angular distance in radian to consider that the rays are closed enough.

    Returns
    -------
    match : torch.Tensor
        The matching rate of shape broadcast(n, n').

    Examples
    --------
    >>> import torch
    >>> from laueimproc.diffraction.metric import compute_matching_rate
    >>> exp_uq = torch.randn(1000, 3)
    >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True)
    >>> theo_uq = torch.randn(5000, 3)
    >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True)
    >>> rate = compute_matching_rate(exp_uq, theo_uq, 0.5 * torch.pi/180)
    >>>
    """
    assert isinstance(exp_uq, torch.Tensor), exp_uq.__class__.__name__
    assert exp_uq.ndim >= 2 and exp_uq.shape[-1] == 3, exp_uq.shape
    assert isinstance(theo_uq, torch.Tensor), theo_uq.__class__.__name__
    assert theo_uq.ndim >= 2 and theo_uq.shape[-1] == 3, theo_uq.shape
    *batch_exp, size_e, _ = exp_uq.shape
    *batch_theo, size_t, _ = theo_uq.shape
    assert size_e * size_t, "can not compute distance of empty matrix"

    if not _no_c and c_metric is not None:
        batch_n = torch.broadcast_shapes(batch_exp, batch_theo)
        exp_uq = torch.broadcast_to(
            exp_uq, batch_n + (size_e, 3)
        ).reshape(int(np.prod(batch_n)), size_e, 3)
        theo_uq = torch.broadcast_to(
            theo_uq, batch_n + (size_t, 3)
        ).reshape(int(np.prod(batch_n)), size_t, 3)
        if threading.current_thread().name == "MainThread":
            with multiprocessing.pool.ThreadPool() as pool:
                rate = pool.starmap(
                    c_metric.matching_rate,
                    (
                        (
                            theo.numpy(force=True),
                            phi_max, exp.numpy(force=True),
                            4.0*phi_max,
                            ray_to_table(exp, 4.0*phi_max),
                        )
                        for theo, exp in zip(theo_uq, exp_uq)
                    ),
                )
        else:
            rate = [
                c_metric.matching_rate(
                    theo.numpy(force=True), phi_max, exp.numpy(force=True), 4.0*phi_max,
                    ray_to_table(exp, 4.0*phi_max),
                )
                for theo, exp in zip(theo_uq, exp_uq)
            ]
        return torch.asarray(rate, dtype=torch.int32, device=theo_uq.device).reshape(batch_n)

    assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
    assert 0 <= phi_max < math.pi, phi_max
    dist = raydotdist(exp_uq, theo_uq)  # (*n, e, t)
    dist = dist > math.cos(phi_max)
    dist = torch.any(dist, dim=-2)  # (*n, t)
    rate = torch.count_nonzero(dist, dim=-1)  # (*n,)
    rate = rate.to(torch.int32)
    return rate


def compute_matching_rate_continuous(
    exp_uq: torch.Tensor, theo_uq: torch.Tensor, phi_max: numbers.Real, *, _no_c: bool = False
) -> torch.Tensor:
    r"""Compute the matching rate.

    This is a continuity extension of the disctere function
    ``laueimproc.diffraction.metric.compute_matching_rate``.
    Let \(\phi\) be the angle between two rays.
    The matching rate is defined with \(r = \sum f(\phi_i)\)
    with \(f(\phi) = e^{-\frac{\log(2)}{\phi_{max}}\phi}\).

    Parameters
    ----------
    exp_uq : torch.Tensor
        The unitary experimental uq vector of shape (*n, e, 3).
    theo_uq : torch.Tensor
        The unitary simulated theorical uq vector of shape (*n', t, 3).
    phi_max : float
        The maximum positive angular distance in radian to consider that the rays are closed enough.

    Returns
    -------
    rate : torch.Tensor
        The continuous matching rate of shape broadcast(n, n').

    Examples
    --------
    >>> import torch
    >>> from laueimproc.diffraction.metric import compute_matching_rate_continuous
    >>> exp_uq = torch.randn(1000, 3)
    >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True)
    >>> theo_uq = torch.randn(5000, 3)
    >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True)
    >>> rate = compute_matching_rate_continuous(exp_uq, theo_uq, 0.5 * torch.pi/180)
    >>>
    """
    assert isinstance(exp_uq, torch.Tensor), exp_uq.__class__.__name__
    assert exp_uq.ndim >= 2 and exp_uq.shape[-1] == 3, exp_uq.shape
    assert isinstance(theo_uq, torch.Tensor), theo_uq.__class__.__name__
    assert theo_uq.ndim >= 2 and theo_uq.shape[-1] == 3, theo_uq.shape
    *batch_exp, size_e, _ = exp_uq.shape
    *batch_theo, size_t, _ = theo_uq.shape
    assert size_e * size_t, "can not compute distance of empty matrix"

    if not _no_c and c_metric is not None:
        batch_n = torch.broadcast_shapes(batch_exp, batch_theo)
        exp_uq = torch.broadcast_to(
            exp_uq, batch_n + (size_e, 3)
        ).reshape(int(np.prod(batch_n)), size_e, 3)
        theo_uq = torch.broadcast_to(
            theo_uq, batch_n + (size_t, 3)
        ).reshape(int(np.prod(batch_n)), size_t, 3)
        if threading.current_thread().name == "MainThread":
            with multiprocessing.pool.ThreadPool() as pool:
                pair_dist = pool.starmap(
                    c_metric.link_close_rays,
                    (
                        (
                            theo.numpy(force=True),
                            phi_max, exp.numpy(force=True),
                            4.0*phi_max,
                            ray_to_table(exp, 4.0*phi_max),
                        )
                        for theo, exp in zip(theo_uq, exp_uq)
                    ),
                )
        else:
            pair_dist = [
                c_metric.link_close_rays(
                    theo.numpy(force=True), phi_max, exp.numpy(force=True), 4.0*phi_max,
                    ray_to_table(exp, 4.0*phi_max),
                )
                for theo, exp in zip(theo_uq, exp_uq)
            ]
        if theo_uq.requires_grad:
            dist = [
                torch.sum(theo[pair[:, 0]] * exp[pair[:, 1]], dim=1)  # sum of scalar product
                for (pair, _), theo, exp in zip(pair_dist, theo_uq, exp_uq)
            ]
        else:
            dist = [torch.from_numpy(d) for _, d in pair_dist]
        dist = [torch.exp(-math.log(2.0)/phi_max * torch.acos(d)).sum().unsqueeze(0) for d in dist]
        return torch.cat(dist).to(dtype=theo_uq.dtype, device=theo_uq.device).reshape(batch_n)

    assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
    assert 0 <= phi_max < math.pi, phi_max
    dist = raydotdist(exp_uq, theo_uq)  # (*n, e, t)
    dist = torch.amax(dist, dim=-2)  # (*n, t)
    dist = torch.exp(-math.log(2.0)/phi_max * torch.acos(dist))
    dist[dist < 0.5] = 0  # (*n, t)
    rate = torch.sum(dist, dim=-1)  # (*n,)
    return rate
