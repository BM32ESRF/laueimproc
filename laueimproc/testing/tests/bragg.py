#!/usr/bin/env python3

"""Test the function of bragg diffraction."""

import torch

from laueimproc.diffraction.bragg import reciprocal_hkl_to_energy, reciprocal_hkl_to_uq


RECIPROCAL = torch.eye(3)  # cubic in Bc
BATCH_RECIPROCAL = (10, 11, 12)
BATCHED_RECIPROCAL = RECIPROCAL[None, None, None, :, :].expand(*BATCH_RECIPROCAL, -1, -1).clone()
HKL = torch.tensor([1, 1, 1])
BATCH_HKL = (13, 14, 15)
BATCHED_HKL = HKL[None, None, None, :].expand(*BATCH_HKL, -1).clone()


def test_batch_reciprocal_hkl_to_energy():
    """Tests batch dimension."""
    assert reciprocal_hkl_to_energy(
        torch.empty(0, 3, 3), torch.empty(0, 3, dtype=int)
    ).shape == (0, 0)
    assert reciprocal_hkl_to_energy(RECIPROCAL, torch.empty(0, 3, dtype=int)).shape == (0,)
    assert reciprocal_hkl_to_energy(torch.empty(0, 3, 3), HKL).shape == (0,)
    assert reciprocal_hkl_to_energy(RECIPROCAL, HKL).shape == ()
    assert reciprocal_hkl_to_energy(
        BATCHED_RECIPROCAL, BATCHED_HKL
    ).shape == BATCH_RECIPROCAL + BATCH_HKL


def test_batch_reciprocal_hkl_to_uq():
    """Tests batch dimension."""
    assert reciprocal_hkl_to_uq(
        torch.empty(0, 3, 3), torch.empty(0, 3, dtype=int)
    ).shape == (0, 0, 3)
    assert reciprocal_hkl_to_uq(RECIPROCAL, torch.empty(0, 3, dtype=int)).shape == (0, 3)
    assert reciprocal_hkl_to_uq(torch.empty(0, 3, 3), HKL).shape == (0, 3)
    assert reciprocal_hkl_to_uq(RECIPROCAL, HKL).shape == (3,)
    assert reciprocal_hkl_to_uq(
        BATCHED_RECIPROCAL, BATCHED_HKL
    ).shape == (*BATCH_RECIPROCAL, *BATCH_HKL, 3)


def test_jac_reciprocal_hkl_to_energy():
    """Tests compute jacobian."""
    assert torch.func.jacrev(reciprocal_hkl_to_energy)(RECIPROCAL, HKL).shape == (3, 3)


def test_jac_reciprocal_hkl_to_uq():
    """Tests compute jacobian."""
    assert torch.func.jacrev(reciprocal_hkl_to_uq)(RECIPROCAL, HKL).shape == (3, 3, 3)
