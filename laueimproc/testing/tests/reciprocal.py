#!/usr/bin/env python3

"""Test the convertion between primitive and reciprocal."""

import torch

from laueimproc.geometry.reciprocal import primitive_to_reciprocal, reciprocal_to_primitive


PRIMITIVE = torch.eye(3)  # cubic in Bc
BATCH_PRIMITIVE = (10, 11, 12)
BATCHED_PRIMITIVE = PRIMITIVE[None, None, None, :, :].expand(*BATCH_PRIMITIVE, -1, -1).clone()


def test_batch_primitive_to_reciprocal():
    """Tests batch dimension."""
    assert primitive_to_reciprocal(torch.empty(0, 3, 3)).shape == (0, 3, 3)
    assert primitive_to_reciprocal(PRIMITIVE).shape == (3, 3)
    assert primitive_to_reciprocal(BATCHED_PRIMITIVE).shape == (*BATCH_PRIMITIVE, 3, 3)


def test_jac_primitive_to_reciprocal():
    """Tests compute jacobian."""
    assert torch.func.jacrev(primitive_to_reciprocal)(PRIMITIVE).shape == (3, 3, 3, 3)


def test_bij():  # inv(f) = f, test only one direction
    """Test is the transformation is reversible."""
    primitive = torch.eye(3, dtype=torch.float64) + torch.randn(1000, 3, 3, dtype=torch.float64) / 3
    primitive_bis = reciprocal_to_primitive(primitive_to_reciprocal(primitive))
    assert torch.allclose(primitive, primitive_bis)
