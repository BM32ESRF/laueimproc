#!/usr/bin/env python3

"""Test if the matching rate is ok."""

import torch

from laueimproc.geometry.metric import compute_matching_rate, compute_matching_rate_continuous


PHI_MAX = 0.5 * torch.pi / 180.0
EXP_UQ = torch.randn(500, 3)
EXP_UQ /= torch.linalg.norm(EXP_UQ, dim=1, keepdims=True)
THEO_UQ = torch.randn(2000, 3)
THEO_UQ /= torch.linalg.norm(THEO_UQ, dim=1, keepdims=True)


def test_batch_matching_rate():
    """Test batch dimension."""
    assert compute_matching_rate(EXP_UQ, THEO_UQ, PHI_MAX, _no_c=True).shape == ()
    assert compute_matching_rate(EXP_UQ, THEO_UQ, PHI_MAX).shape == ()
    exp_uq = EXP_UQ[None, None, None, :, :].expand(1, 3, 4, -1, -1)
    theo_uq = THEO_UQ[None, None, None, :, :].expand(2, 1, 4, -1, -1)
    assert compute_matching_rate(exp_uq, theo_uq, PHI_MAX, _no_c=True).shape == (2, 3, 4)
    assert compute_matching_rate(exp_uq, theo_uq, PHI_MAX).shape == (2, 3, 4)


def test_batch_matching_rate_continuous():
    """Test batch dimension."""
    assert compute_matching_rate_continuous(EXP_UQ, THEO_UQ, PHI_MAX, _no_c=True).shape == ()
    assert compute_matching_rate_continuous(EXP_UQ, THEO_UQ, PHI_MAX).shape == ()
    exp_uq = EXP_UQ[None, None, None, :, :].expand(1, 3, 4, -1, -1)
    theo_uq = THEO_UQ[None, None, None, :, :].expand(2, 1, 4, -1, -1)
    assert compute_matching_rate_continuous(exp_uq, theo_uq, PHI_MAX, _no_c=True).shape == (2, 3, 4)
    assert compute_matching_rate_continuous(exp_uq, theo_uq, PHI_MAX).shape == (2, 3, 4)


def test_grad_matching_rate_continuous():
    """Test the continuous matching rate is differenciable."""
    theo_uq = THEO_UQ.clone()
    theo_uq.requires_grad = True
    rate = compute_matching_rate_continuous(EXP_UQ, theo_uq, PHI_MAX)
    rate.backward()
    assert theo_uq.grad is not None


def test_value_matching_rate():
    """Test if the result of the matching rate is correct."""
    assert int(compute_matching_rate(EXP_UQ, EXP_UQ, PHI_MAX, _no_c=True)) == len(EXP_UQ)
    assert int(compute_matching_rate(EXP_UQ, EXP_UQ, PHI_MAX)) == len(EXP_UQ)
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
