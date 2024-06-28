#!/usr/bin/env python3

"""Implement some loss functions."""

import math
import numbers

import torch

from .projection import detector_to_ray


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
        The 2d point associated to uf in the referencial of the detector of shape (*n, r2, 2)).
        Could be directly the unitary ray vector uf or uq of shape (*n, r2, 3).
    poni : torch.Tensor, optional
        Only use if the ray are projected points.
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (*p, 6).

    Returns
    -------
    dist : torch.Tensor
        The distance matrix \(\cos(\phi)\) of shape (*n, *p, r1, r2).
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
    exp_uq: torch.Tensor, theo_uq: torch.Tensor, phi_max: numbers.Real
) -> torch.Tensor:
    r"""Compute the matching rate.

    Parameters
    ----------
    exp_uq : torch.Tensor
        The unitary experimental uq vector of shape (*n, e, 3).
    theo_uq : torch.Tensor
        The unitary simulated theorical uq vector of shape (*n, t, 3).
    phi_max : float
        The maximum positive angular distance in radian to consider that the rays are closed enough.

    Returns
    -------
    match : torch.Tensor
        The matching rate of shape n.
    """
    assert isinstance(exp_uq, torch.Tensor), exp_uq.__class__.__name__
    assert exp_uq.ndim >= 2 and exp_uq.shape[-1] == 3, exp_uq.shape
    assert isinstance(theo_uq, torch.Tensor), theo_uq.__class__.__name__
    assert theo_uq.ndim >= 2 and theo_uq.shape[-1] == 3, theo_uq.shape
    assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
    assert 0 <= phi_max < math.pi/2, phi_max

    dist = raydotdist(exp_uq, theo_uq)  # (*n, e, t)

    dist = dist > math.cos(phi_max)
    dist = torch.any(dist, dim=-2)  # (*n, t)
    rate = torch.count_nonzero(dist, dim=-1)  # (*n,)
    rate = rate.to(theo_uq.dtype)
    # rate /= exp_uq.shape[-2]
    return rate


# def compute_dist(x, y):
#     diff = x[..., *((None,)*(y.ndim-1)), :] - y[*((None,)*(x.ndim-1)), ..., :]  # (*x, *y, 2)
#     return torch.sqrt(torch.sum(diff * diff, dim=-1))


# def couple(ref, x, eps=0.1):
#     # make reference hash table
#     ref_dict = {}  # to each rounded couple (x, y) associate the ref indices
#     for i, idx in enumerate((0.5 + ref/eps).to(torch.int64).tolist()):
#         idx = tuple(idx)
#         ref_dict[idx] = ref_dict.get(idx, [])
#         ref_dict[idx].append(i)

#     # compare each element
#     quandidates = []  # to each dst indicies, associate the ref candidates indicies
#     for i, (x1_floor, x2_floor) in enumerate((x/eps).to(torch.int64).tolist()):
#         quandidates.append(
#             ref_dict.get((x1_floor, x2_floor), [])
#             + ref_dict.get((x1_floor, x2_floor + 1), [])
#             + ref_dict.get((x1_floor + 1, x2_floor), [])
#             + ref_dict.get((x1_floor + 1, x2_floor + 1), [])
#         )

#     # each matrix distances
#     distances = []  # to each dst indices, associate the distance with all the ref candidates
#     for i_x, i_refs in enumerate(quandidates):
#         distances.append(compute_dist(x[i_x], ref[i_refs]))
