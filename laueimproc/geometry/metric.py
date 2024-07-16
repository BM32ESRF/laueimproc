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
    from laueimproc.geometry import c_metric
except ImportError:
    logging.warning(
        "failed to import laueimproc.geometry.c_metric, a slow python version is used instead"
    )
    c_metric = None
from .projection import detector_to_ray


def _cached_ray_to_table(func):
    """Decorate ray_to_table to cache the result."""
    @functools.wraps(func)
    def decorated(
        ray: np.ndarray, res: numbers.Real
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """Invoque ``c_metric.ray_to_table`` and keep result in cache.

        Parameters
        ----------
        ray : np.ndarray
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
        assert isinstance(ray, np.ndarray), ray.__class__.__name__
        signature = (ray.ctypes.data, ray.shape[0], float(res))
        if signature not in cache:
            assert ray.ndim == 2 and ray.shape[1] == 3, ray.shape  # not a head for perfs
            if len(cache) > 64:  # to avoid memory leaking
                cache.clear()
            cache[signature] = c_metric.ray_to_table(ray, res)
        return cache[signature]
    cache = {}
    return decorated


ray_to_table = None if c_metric is None else _cached_ray_to_table(c_metric.ray_to_table)


def ray_cosine_dist(
    ray_point_1: torch.Tensor, ray_point_2: torch.Tensor, poni: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Compute the scalar product matrix of the rays pairwise.

    Parameters
    ----------
    ray_point_1 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (\*n, r1, 2).
        Could be directly the unitary ray vector uf or uq of shape (\*n, r1, 3).
    ray_point_2 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (\*n', r2, 2)).
        Could be directly the unitary ray vector uf or uq of shape (\*n', r2, 3).
    poni : torch.Tensor, optional
        Only use if the ray are projected points.
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\*p, 6).

    Returns
    -------
    dist : torch.Tensor
        The distance matrix \(\cos(\phi)\) of shape (\*broadcast(n, n'), \*p, r1, r2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.metric import ray_cosine_dist
    >>> ray_1 = torch.randn(500, 3)
    >>> ray_1 /= torch.linalg.norm(ray_1, dim=1, keepdims=True)
    >>> ray_2 = torch.randn(2000, 3)
    >>> ray_2 /= torch.linalg.norm(ray_2, dim=1, keepdims=True)
    >>> ray_cosine_dist(ray_1, ray_2).shape
    torch.Size([500, 2000])
    >>>
    """
    assert isinstance(ray_point_1, torch.Tensor), ray_point_1.__class__.__name__
    assert isinstance(ray_point_2, torch.Tensor), ray_point_2.__class__.__name__

    if ray_point_1.shape[-1] == 2:
        assert poni is not None
        ray_1 = detector_to_ray(ray_point_1, poni)  # (*broadcast(p, p'), *r1, 3)
    elif ray_point_1.shape[-1] == 3:
        ray_1 = ray_point_1
    else:
        raise ValueError("only shape 2 and 3 allow")
    if ray_point_2.shape[-1] == 2:
        assert poni is not None
        ray_2 = detector_to_ray(ray_point_2, poni)  # (*broadcast(p, p'), *r2, 3)
    elif ray_point_2.shape[-1] == 3:
        ray_2 = ray_point_2
    else:
        raise ValueError("only shape 2 and 3 allow")

    ray_1 = ray_1[..., :, None, :]
    ray_2 = ray_2[..., None, :, :]
    dist = torch.sum(ray_1 * ray_2, dim=-1)  # <ray_1, ray_2> = cos(angle(ray_1, ray_2)) = cos(phi)
    return dist


def ray_phi_dist(
    ray_point_1: torch.Tensor, ray_point_2: torch.Tensor, poni: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Compute the angle distance matrix of the rays pairwise.

    Parameters
    ----------
    ray_point_1 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (\*n, r1, 2).
        Could be directly the unitary ray vector uf or uq of shape (\*n, r1, 3).
    ray_point_2 : torch.Tensor
        The 2d point associated to uf in the referencial of the detector of shape (\*n', r2, 2)).
        Could be directly the unitary ray vector uf or uq of shape (\*n', r2, 3).
    poni : torch.Tensor, optional
        Only use if the ray are projected points.
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\*p, 6).

    Returns
    -------
    phi : torch.Tensor
        The distance matrix \(\phi\) of shape (\*broadcast(n, n'), \*p, r1, r2).
        \(\phi \in [0, \pi]\).

    Notes
    -----
    It's mathematically equivalent to calculating acos of ``laueimproc.geometry.metric.``,
    but this function is numerically more stable for small angles.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.metric import ray_phi_dist
    >>> ray_1 = torch.randn(500, 3)
    >>> ray_1 /= torch.linalg.norm(ray_1, dim=1, keepdims=True)
    >>> ray_2 = torch.randn(2000, 3)
    >>> ray_2 /= torch.linalg.norm(ray_2, dim=1, keepdims=True)
    >>> ray_phi_dist(ray_1, ray_2).shape
    torch.Size([500, 2000])
    >>>
    """
    assert isinstance(ray_point_1, torch.Tensor), ray_point_1.__class__.__name__
    assert isinstance(ray_point_2, torch.Tensor), ray_point_2.__class__.__name__

    if ray_point_1.shape[-1] == 2:
        assert poni is not None
        ray_1 = detector_to_ray(ray_point_1, poni)  # (*broadcast(p, p'), *r1, 3)
    elif ray_point_1.shape[-1] == 3:
        ray_1 = ray_point_1
    else:
        raise ValueError("only shape 2 and 3 allow")
    if ray_point_2.shape[-1] == 2:
        assert poni is not None
        ray_2 = detector_to_ray(ray_point_2, poni)  # (*broadcast(p, p'), *r2, 3)
    elif ray_point_2.shape[-1] == 3:
        ray_2 = ray_point_2
    else:
        raise ValueError("only shape 2 and 3 allow")

    ray_1 = ray_1[..., :, None, :]
    ray_2 = ray_2[..., None, :, :]

    cosine_dist = torch.sum(ray_1 * ray_2, dim=-1)

    mask_p1 = cosine_dist > 0.707
    mask_m1 = cosine_dist < -0.707
    ray_1, ray_2 = torch.broadcast_tensors(ray_1, ray_2)
    cross_p1 = torch.sqrt(torch.sum(torch.linalg.cross(ray_1[mask_p1], ray_2[mask_p1])**2, dim=-1))
    cross_m1 = torch.sqrt(torch.sum(torch.linalg.cross(ray_1[mask_m1], ray_2[mask_m1])**2, dim=-1))
    phi_p1 = torch.asin(cross_p1)
    phi_m1 = torch.pi - torch.asin(cross_m1)

    phi = torch.empty_like(cosine_dist)
    phi[~(mask_p1 | mask_m1)] = torch.acos(cosine_dist[~(mask_p1 | mask_m1)])
    phi[mask_p1] = phi_p1
    phi[mask_m1] = phi_m1
    return phi


def compute_matching_rate(
    uq_exp: torch.Tensor, uq_theo: torch.Tensor, phi_max: numbers.Real, *, _no_c: bool = False
) -> torch.Tensor:
    r"""Compute the matching rate.

    It is the number of ray in ``uq_theo``, close engouth to at least one ray of ``uq_exp``.

    Parameters
    ----------
    uq_exp : torch.Tensor
        The unitary experimental uq vector of shape (\*n, e, 3).
    uq_theo : torch.Tensor
        The unitary simulated theorical uq vector of shape (\*n', t, 3).
        It can be in any device.
    phi_max : float
        The maximum positive angular distance in radian to consider that the rays are closed enough.

    Returns
    -------
    match : torch.Tensor
        The matching rate of shape broadcast(n, n') in the ``uq_exp`` device.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.metric import compute_matching_rate
    >>> uq_exp = torch.randn(1000, 3)
    >>> uq_exp /= torch.linalg.norm(uq_exp, dim=1, keepdims=True)
    >>> uq_theo = torch.randn(5000, 3)
    >>> uq_theo /= torch.linalg.norm(uq_theo, dim=1, keepdims=True)
    >>> rate = compute_matching_rate(uq_exp, uq_theo, 0.5 * torch.pi/180)
    >>>
    """
    assert isinstance(uq_exp, torch.Tensor), uq_exp.__class__.__name__
    assert uq_exp.ndim >= 2 and uq_exp.shape[-1] == 3, uq_exp.shape
    assert isinstance(uq_theo, torch.Tensor), uq_theo.__class__.__name__
    assert uq_theo.ndim >= 2 and uq_theo.shape[-1] == 3, uq_theo.shape
    *batch_exp, size_e, _ = uq_exp.shape
    *batch_theo, size_t, _ = uq_theo.shape
    assert size_e * size_t, "can not compute distance of empty matrix"

    if not _no_c and c_metric is not None:
        batch_n = torch.broadcast_shapes(batch_exp, batch_theo)
        uq_exp = torch.broadcast_to(
            uq_exp, batch_n + (size_e, 3)
        ).reshape(int(np.prod(batch_n)), size_e, 3)
        uq_theo = torch.broadcast_to(
            uq_theo, batch_n + (size_t, 3)
        ).reshape(int(np.prod(batch_n)), size_t, 3)
        if threading.current_thread().name == "MainThread":
            with multiprocessing.pool.ThreadPool() as pool:
                rate = pool.starmap(
                    c_metric.matching_rate,
                    (
                        (theo, phi_max, exp, 4.0*phi_max, ray_to_table(exp, 4.0*phi_max))
                        for theo, exp in zip(uq_theo.numpy(force=True), uq_exp.numpy(force=True))
                    ),
                )
        else:
            rate = [
                c_metric.matching_rate(
                    theo, phi_max, exp, 4.0*phi_max, ray_to_table(exp, 4.0*phi_max),
                )
                for theo, exp in zip(uq_theo.numpy(force=True), uq_exp.numpy(force=True))
            ]
        return torch.asarray(rate, dtype=torch.int32, device=uq_exp.device).reshape(batch_n)

    assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
    assert 0 <= phi_max < math.pi, phi_max
    dist = ray_cosine_dist(uq_exp, uq_theo.to(uq_exp.device))  # (*n, e, t)
    dist = dist > math.cos(phi_max)
    dist = torch.any(dist, dim=-2)  # (*n, t)
    rate = torch.count_nonzero(dist, dim=-1)  # (*n,)
    rate = rate.to(torch.int32)
    return rate


def compute_matching_rate_continuous(
    uq_exp: torch.Tensor, uq_theo: torch.Tensor, phi_max: numbers.Real, *, _no_c: bool = False
) -> torch.Tensor:
    r"""Compute the matching rate.

    This is a continuity extension of the disctere function
    ``laueimproc.geometry.metric.compute_matching_rate``.
    Let \(\phi\) be the angle between two rays.
    The matching rate is defined with \(r = \sum f(\phi_i)\)
    with \(f(\phi) = e^{-\frac{\log(2)}{\phi_{max}}\phi}\).

    Parameters
    ----------
    uq_exp : torch.Tensor
        The unitary experimental uq vector of shape (\*n, e, 3).
    uq_theo : torch.Tensor
        The unitary simulated theorical uq vector of shape (\*n', t, 3).
    phi_max : float
        The maximum positive angular distance in radian to consider that the rays are closed enough.

    Returns
    -------
    rate : torch.Tensor
        The continuous matching rate of shape broadcast(n, n').

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.metric import compute_matching_rate_continuous
    >>> uq_exp = torch.randn(1000, 3)
    >>> uq_exp /= torch.linalg.norm(uq_exp, dim=1, keepdims=True)
    >>> uq_theo = torch.randn(5000, 3)
    >>> uq_theo /= torch.linalg.norm(uq_theo, dim=1, keepdims=True)
    >>> rate = compute_matching_rate_continuous(uq_exp, uq_theo, 0.5 * torch.pi/180)
    >>>
    """
    assert isinstance(uq_exp, torch.Tensor), uq_exp.__class__.__name__
    assert uq_exp.ndim >= 2 and uq_exp.shape[-1] == 3, uq_exp.shape
    assert isinstance(uq_theo, torch.Tensor), uq_theo.__class__.__name__
    assert uq_theo.ndim >= 2 and uq_theo.shape[-1] == 3, uq_theo.shape
    *batch_exp, size_e, _ = uq_exp.shape
    *batch_theo, size_t, _ = uq_theo.shape
    assert size_e * size_t, "can not compute distance of empty matrix"

    factor = -math.log(2.0) / phi_max

    if not _no_c and c_metric is not None:
        batch_n = torch.broadcast_shapes(batch_exp, batch_theo)
        uq_exp = torch.broadcast_to(
            uq_exp, batch_n + (size_e, 3)
        ).reshape(int(np.prod(batch_n)), size_e, 3)
        uq_theo = torch.broadcast_to(
            uq_theo, batch_n + (size_t, 3)
        ).reshape(int(np.prod(batch_n)), size_t, 3)
        if threading.current_thread().name == "MainThread":
            with multiprocessing.pool.ThreadPool() as pool:
                pairs = pool.starmap(
                    c_metric.link_close_rays,
                    (
                        (theo, phi_max, exp, 4.0*phi_max, ray_to_table(exp, 4.0*phi_max))
                        for theo, exp in zip(
                            uq_theo.to(torch.float32).numpy(force=True),
                            uq_exp.to(torch.float32).numpy(force=True),
                        )
                    ),
                )
        else:
            pairs = [
                c_metric.link_close_rays(
                    theo, phi_max, exp, 4.0*phi_max, ray_to_table(exp, 4.0*phi_max)
                )
                for theo, exp in zip(
                    uq_theo.to(torch.float32).numpy(force=True),
                    uq_exp.to(torch.float32).numpy(force=True),
                )
            ]
        dists = [
            torch.exp(factor * ray_phi_dist(
                uq_exp[i, pair[:, 1], None], uq_theo[i, pair[:, 0], None]
            )[:, 0, 0]) for i, pair in enumerate(pairs)
        ]
        return torch.cat([  # dist[dist >= 0.5].sum() failed if no matching
            dist.where(dist > 0.5, 0.0).sum().unsqueeze(0) for dist in dists
        ]).reshape(batch_n)

    assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
    assert 0 <= phi_max < math.pi, phi_max
    phi = ray_phi_dist(uq_exp, uq_theo)  # (*n, e, t)
    phi = torch.amin(phi, dim=-2)  # (*n, t)
    dist = torch.exp(factor * phi)
    dist = dist.where(dist > 0.5, 0.0)  # (*n, t), "dist[dist < 0.5] = 0.0" failed backward
    rate = torch.sum(dist, dim=-1)  # (*n,)
    return rate
