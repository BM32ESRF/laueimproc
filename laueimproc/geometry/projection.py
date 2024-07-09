#!/usr/bin/env python3

"""Project rays on a physical or virtual plane."""


import torch

from .rotation import angle_to_rot


def detector_to_ray(point: torch.Tensor, poni: torch.Tensor) -> torch.Tensor:
    r"""Find light ray witch intersected on the detector.

    Parameters
    ----------
    point : torch.Tensor
        The 2d point in the referencial of the detector of shape (\*r, \*p, 2).
    poni : torch.Tensor
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\*p', 6).

    Returns
    -------
    ray : torch.Tensor
        The unitary ray vector of shape (\*r, \*broadcast(p, p'), 3).
    """
    assert isinstance(point, torch.Tensor), point.__class__.__name__
    assert point.shape[-1:] == (2,), point.shape
    assert isinstance(poni, torch.Tensor), poni.__class__.__name__
    assert poni.shape[-1:] == (6,), poni.shape

    *batch, _ = point.shape
    *batch_poni, _ = poni.shape
    batch_ray = batch[:len(batch)-len(batch_poni)]
    batch_poni = torch.broadcast_shapes(batch_poni, batch[len(batch)-len(batch_poni):])

    point = point - poni[*((None,) * len(batch_ray)), ..., 1:3]  # detector shift
    ray = torch.empty(*batch_ray, *batch_poni, 3, dtype=point.dtype, device=point.device)
    ray[..., :2] = point
    ray[..., 2] = poni[*((None,) * len(batch_ray)), ..., 0]
    rot = angle_to_rot(poni[..., 3], poni[..., 4], -poni[..., 5], meshgrid=False)
    rot = rot[*((None,) * len(batch_ray)), ..., :, :]
    ray = (rot.mT @ ray[..., None]).squeeze(-1)
    ray = ray * torch.rsqrt(torch.sum(ray * ray, dim=-1, keepdim=True))
    return ray


def ray_to_detector(ray: torch.Tensor, poni: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the intersection of the light ray with the detector.

    Parameters
    ----------
    ray : torch.Tensor
        The unitary ray vector of shape (*r, 3).
    poni : torch.Tensor
        The point of normal incidence, callibration parameters according the pyfai convention.
        Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (*p, 6).

    Returns
    -------
    point : torch.Tensor
        The 2d points in the referencial of the detector of shape (*r, *p, 2).
    dist : torch.Tensor
        The algebrical distance of the ray between the sample and the detector.
        a positive value means that the beam crashs on the detector.
        A negative value means it is moving away.
        The shape is (*r, *p).
    """
    assert isinstance(ray, torch.Tensor), ray.__class__.__name__
    assert ray.shape[-1:] == (3,), ray.shape
    assert isinstance(poni, torch.Tensor), poni.__class__.__name__
    assert poni.shape[-1:] == (6,), poni.shape

    rot = angle_to_rot(poni[..., 3], poni[..., 4], -poni[..., 5], meshgrid=False)  # (*p, 3, 3)
    rectified_ray = (
        rot[*((None,) * (ray.ndim - 1)), ..., :, :]
        @ ray[..., *((None,) * (poni.ndim - 1)), :, None]
    ).squeeze(-1)  # (*p, *r, 3)
    ray_dist = poni[*((None,) * (ray.ndim - 1)), ..., 0] / rectified_ray[..., 2]
    point = ray_dist.unsqueeze(-1) * rectified_ray[..., :2]
    point = point + poni[*((None,) * (ray.ndim - 1)), ..., 1:3]  # detector shift

    return point, ray_dist
