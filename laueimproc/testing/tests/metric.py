#!/usr/bin/env python3

"""Test if the matching rate is ok."""

import torch

from laueimproc.geometry.metric import (
    compute_matching_rate, compute_matching_rate_continuous, ray_cosine_dist, ray_phi_dist
)


PHI_MAX = 0.5 * torch.pi/180
UQ_EXP = torch.randn(500, 3)
UQ_EXP /= torch.linalg.norm(UQ_EXP, dim=1, keepdims=True)
UQ_THEO = torch.randn(2000, 3)
UQ_THEO /= torch.linalg.norm(UQ_THEO, dim=1, keepdims=True)


def test_batch_matching_rate():
    """Test batch dimension."""
    assert compute_matching_rate(UQ_EXP, UQ_THEO, PHI_MAX, _no_c=True).shape == ()
    assert compute_matching_rate(UQ_EXP, UQ_THEO, PHI_MAX).shape == ()
    uq_exp = UQ_EXP[None, None, None, :, :].expand(1, 3, 4, -1, -1)
    uq_theo = UQ_THEO[None, None, None, :, :].expand(2, 1, 4, -1, -1)
    assert compute_matching_rate(uq_exp, uq_theo, PHI_MAX, _no_c=True).shape == (2, 3, 4)
    assert compute_matching_rate(uq_exp, uq_theo, PHI_MAX).shape == (2, 3, 4)


def test_batch_matching_rate_continuous():
    """Test batch dimension."""
    assert compute_matching_rate_continuous(UQ_EXP, UQ_THEO, PHI_MAX, _no_c=True).shape == ()
    assert compute_matching_rate_continuous(UQ_EXP, UQ_THEO, PHI_MAX).shape == ()
    uq_exp = UQ_EXP[None, None, None, :, :].expand(1, 3, 4, -1, -1)
    uq_theo = UQ_THEO[None, None, None, :, :].expand(2, 1, 4, -1, -1)
    assert compute_matching_rate_continuous(uq_exp, uq_theo, PHI_MAX, _no_c=True).shape == (2, 3, 4)
    assert compute_matching_rate_continuous(uq_exp, uq_theo, PHI_MAX).shape == (2, 3, 4)


def test_cosine_eq_phi():
    """Test the cosine dist gives the same results as the phi dist."""
    uq_exp = torch.randn(500, 3, dtype=torch.float64)
    uq_exp /= torch.linalg.norm(uq_exp, dim=1, keepdims=True)
    uq_theo = torch.randn(2000, 3, dtype=torch.float64)
    uq_theo /= torch.linalg.norm(uq_theo, dim=1, keepdims=True)
    phi_1 = torch.acos(ray_cosine_dist(uq_exp, uq_theo))
    phi_2 = ray_phi_dist(uq_exp, uq_theo)
    assert torch.allclose(phi_1, phi_2)


def test_grad_matching_rate_continuous():
    """Test the continuous matching rate is differenciable."""
    uq_theo = UQ_THEO.clone()
    uq_theo.requires_grad = True
    rate = compute_matching_rate_continuous(UQ_EXP, uq_theo, PHI_MAX)
    rate.backward()
    assert not uq_theo.grad.isnan().any()


# def test_grad_phi_dist():
#     """Test the back propagation is ok."""
#     uq_exp = UQ_THEO.clone()
#     uq_exp.requires_grad = True
#     uq_theo = UQ_THEO.clone()
#     uq_theo.requires_grad = True
#     phi = ray_phi_dist(uq_exp, uq_theo)
#     phi.sum().backward()
#     assert not uq_exp.grad.isnan().any()
#     assert not uq_theo.grad.isnan().any()


def test_value_matching_rate():
    """Test if the result of the matching rate is correct."""
    assert int(compute_matching_rate(UQ_EXP, UQ_EXP, PHI_MAX, _no_c=True)) == len(UQ_EXP)
    assert int(compute_matching_rate(UQ_EXP, UQ_EXP, PHI_MAX)) == len(UQ_EXP)
    assert int(compute_matching_rate(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), 0.6 * torch.pi, _no_c=True
    )) == 1
    assert int(compute_matching_rate(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), 0.6 * torch.pi
    )) == 1
    assert int(compute_matching_rate(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), 0.4 * torch.pi, _no_c=True
    )) == 0
    assert int(compute_matching_rate(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), 0.4 * torch.pi
    )) == 0
