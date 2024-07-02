#!/usr/bin/env python3

"""Test if the matching rate is ok."""

import torch

from laueimproc.diffraction.metric import compute_matching_rate


PHI_MAX = 0.5 * torch.pi / 180.0
EXP_UQ = torch.randn(500, 3)
EXP_UQ /= torch.linalg.norm(EXP_UQ, dim=1, keepdims=True)
THEO_UQ = torch.randn(2000, 3)
THEO_UQ /= torch.linalg.norm(THEO_UQ, dim=1, keepdims=True)


def test_batch_matching_rate():
    """Tests batch dimension."""
    assert compute_matching_rate(EXP_UQ, THEO_UQ, PHI_MAX).shape == ()
    assert compute_matching_rate(EXP_UQ, THEO_UQ, PHI_MAX, _no_c=True).shape == ()
    exp_uq = EXP_UQ[None, None, None, :, :].expand(1, 3, 4, -1, -1)
    theo_uq = THEO_UQ[None, None, None, :, :].expand(2, 1, 4, -1, -1)
    assert compute_matching_rate(exp_uq, theo_uq, PHI_MAX).shape == (2, 3, 4)
    assert compute_matching_rate(exp_uq, theo_uq, PHI_MAX, _no_c=True).shape == (2, 3, 4)
