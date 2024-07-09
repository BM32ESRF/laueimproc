#!/usr/bin/env python3

"""Test the convertion between lattice and primitive space."""

import torch

from laueimproc.geometry.lattice import lattice_to_primitive, primitive_to_lattice

LATTICE = torch.tensor([1.0, 1.0, 1.0, torch.pi/2, torch.pi/2, torch.pi/2])  # cubic
BATCH_LATTICE = (7, 8, 9)
BATCHED_LATTICE = LATTICE[None, None, None, :].expand(*BATCH_LATTICE, -1).clone()
PRIMITIVE = torch.eye(3)  # cubic in Bc
BATCH_PRIMITIVE = (10, 11, 12)
BATCHED_PRIMITIVE = PRIMITIVE[None, None, None, :, :].expand(*BATCH_PRIMITIVE, -1, -1).clone()


def test_batch_lattice_to_primitive():
    """Tests batch dimension."""
    assert lattice_to_primitive(torch.empty(0, 6)).shape == (0, 3, 3)
    assert lattice_to_primitive(LATTICE).shape == (3, 3)
    assert lattice_to_primitive(BATCHED_LATTICE).shape == (*BATCH_LATTICE, 3, 3)


def test_jac_lattice_to_primitive():
    """Tests compute jacobian."""
    assert torch.func.jacrev(lattice_to_primitive)(LATTICE).shape == (3, 3, 6)


def test_batch_primitive_to_lattice():
    """Tests batch dimension."""
    assert primitive_to_lattice(torch.empty(0, 3, 3)).shape == (0, 6)
    assert primitive_to_lattice(PRIMITIVE).shape == (6,)
    assert primitive_to_lattice(BATCHED_PRIMITIVE).shape == (*BATCH_PRIMITIVE, 6)


def test_jac_primitive_to_lattice():
    """Tests compute jacobian."""
    assert torch.func.jacrev(primitive_to_lattice)(PRIMITIVE).shape == (6, 3, 3)


def test_bij_lattice_to_primitive_to_lattice():
    """Test is the transformation is reverible."""
    lattice = torch.empty(1000, 6, dtype=torch.float64)
    lattice[..., :3] = (1.0 - 0.2) * torch.rand(len(lattice), 3) + 0.1
    lattice[..., 4:] = (1.0 - 0.2) * torch.pi * torch.rand(len(lattice), 2) + 0.1
    alpha_max = torch.acos(torch.cos(lattice[..., 4]) * torch.cos(lattice[..., 5]))
    alpha_min = torch.acos(
        torch.cos(lattice[..., 4]) * torch.cos(lattice[..., 5])
        + torch.sin(lattice[..., 4]) * torch.sin(lattice[..., 5])
    )
    lattice[..., 3] = (
        (1.0 - 0.2) * (alpha_max - alpha_min) * torch.rand(len(lattice))
        + 0.1 + alpha_min
    )
    lattice_bis = primitive_to_lattice(lattice_to_primitive(lattice))
    assert torch.allclose(lattice, lattice_bis)
