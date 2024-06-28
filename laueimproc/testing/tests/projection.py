#!/usr/bin/env python3

"""Test the convertion between ray and detector position."""


import torch

from laueimproc.diffraction.projection import detector_to_ray, ray_to_detector


RAY = torch.tensor([0.0, 0.0, 1.0])
BATCH_RAY = (22, 23, 24)
BATCHED_RAY = RAY[None, None, None, :].expand(*BATCH_RAY, -1).clone()
POINT = torch.tensor([0.0, 0.0])
BATCH_POINT = (25, 26)
BATCHED_POINT = POINT[None, None, :].expand(*BATCH_POINT, -1).clone()
PONI = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BATCH_PONI = (27, 28)
BATCHED_PONI = PONI[None, None, :].expand(*BATCH_PONI, -1).clone()


def test_batch_detector_to_ray():
    """Tests batch dimension."""
    assert detector_to_ray(torch.empty(0, 2), torch.empty(0, 6)).shape == (0, 3)
    assert detector_to_ray(POINT, PONI).shape == (3,)
    assert (
        detector_to_ray(BATCHED_POINT[None, None, ...], BATCHED_PONI).shape
        == (*BATCH_PONI, *BATCH_POINT, 3)
    )


def test_batch_ray_to_detector():
    """Tests batch dimension."""
    point, dist = ray_to_detector(torch.empty(0, 3), torch.empty(0, 6))
    assert point.shape == (0, 0, 2)
    assert dist.shape == (0, 0)
    point, dist = ray_to_detector(RAY, PONI)
    assert point.shape == (2,)
    assert dist.shape == ()
    point, dist = ray_to_detector(BATCHED_RAY, BATCHED_PONI)
    assert point.shape == (*BATCH_RAY, *BATCH_PONI, 2)
    assert dist.shape == BATCH_RAY + BATCH_PONI


def test_bij_ray_to_point_to_ray():
    """Test ray -> point -> ray = ray."""
    ray = torch.randn(1000, 3, dtype=torch.float64)
    ray *= torch.rsqrt(torch.sum(ray * ray, dim=-1, keepdim=True))
    poni = torch.empty(1000, 6, dtype=torch.float64)
    poni[..., 0] = torch.rand(1000)
    poni[..., 1:3] = torch.randn(1000, 2)
    poni[..., 3:] = 2*torch.pi*torch.rand(1000, 3) - torch.pi

    point, dist = ray_to_detector(ray, poni)
    ray_bis = detector_to_ray(point, poni)
    ray_bis *= torch.sign(dist).unsqueeze(-1)
    ray = ray[:, None, :].expand(-1, 1000, -1)

    assert torch.allclose(ray, ray_bis)


def test_jac_detector_to_ray():
    """Tests compute jacobian."""
    assert torch.func.jacrev(detector_to_ray, 0)(POINT, PONI).shape == (3, 2)
    assert torch.func.jacrev(detector_to_ray, 1)(POINT, PONI).shape == (3, 6)


def test_jac_ray_to_detector():
    """Tests compute jacobian."""
    point_jac, dist_jac = torch.func.jacrev(ray_to_detector, 0)(RAY, PONI)
    assert point_jac.shape == (2, 3)
    assert dist_jac.shape == (3,)
    point_jac, dist_jac = torch.func.jacrev(ray_to_detector, 1)(RAY, PONI)
    assert point_jac.shape == (2, 6)
    assert dist_jac.shape == (6,)


def test_normalization_detector_to_ray():
    """Test norm is 1."""
    point = torch.randn(1, 1000, 2)
    poni = torch.empty(1000, 6)
    poni[..., 0] = torch.rand(1000)
    poni[..., 1:3] = torch.randn(1000, 2)
    poni[..., 3:] = 2*torch.pi*torch.rand(1000, 3) - torch.pi
    ray = detector_to_ray(point, poni)
    assert torch.allclose(torch.linalg.vector_norm(ray, dim=-1), torch.tensor(1.0))
