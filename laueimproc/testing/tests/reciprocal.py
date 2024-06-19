#!/usr/bin/env python3

"""Test the convertion between primitive and reciprocal space."""

import torch

from laueimproc.diffraction.reciprocal import lattice_to_primitive, primitive_to_lattice

LATTICE = torch.tensor([1.0, 1.0, 1.0, torch.pi/2, torch.pi/2, torch.pi/2])  # cubic
BATCH_LATTICE = (7, 8, 9)
BATCHED_LATTICE = LATTICE[None, None, None, :].expand(*BATCH_LATTICE, -1).clone()
PRIMITIVE = torch.eye(3)  # cubic in Bc
BATCH_PRIMITIVE = (10, 11, 12)
BATCHED_PRIMITIVE = PRIMITIVE[None, None, None, :, :].expand(*BATCH_PRIMITIVE, -1, -1).clone()


def test_batch_lattice_to_primitive():
    """Tests batch dimension."""
    assert lattice_to_primitive(LATTICE).shape == (3, 3)
    assert lattice_to_primitive(BATCHED_LATTICE).shape == (*BATCH_LATTICE, 3, 3)


def test_jac_lattice_to_primitive():
    """Tests compute jacobian."""
    assert torch.func.jacrev(lattice_to_primitive)(LATTICE).shape == (3, 3, 6)


def test_batch_primitive_to_lattice():
    """Tests batch dimension."""
    assert primitive_to_lattice(PRIMITIVE).shape == (6,)
    assert primitive_to_lattice(BATCHED_PRIMITIVE).shape == (*BATCH_PRIMITIVE, 6)
