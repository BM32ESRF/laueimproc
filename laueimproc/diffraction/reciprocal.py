#!/usr/bin/env python3

r"""Enables communication between direct \(\mathbf{A}\) and reciprocal space \(\mathbf{B}\)."""

import torch


def primitive_to_reciprocal(primitive: torch.Tensor) -> torch.Tensor:
    r"""Convert the primitive vectors into the reciprocal base vectors.

    Parameters
    ----------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) in any orthonormal base.

    Returns
    -------
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) in the same orthonormal base.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.diffraction.reciprocal import primitive_to_reciprocal
    >>>
    """
    assert isinstance(primitive, torch.Tensor), primitive.__class__.__name__
    assert primitive.shape[-2:] == (3, 3), primitive.shape

    reciprocal = torch.empty_like(primitive)

    volume = torch.linalg.det(primitive).unsqueeze(-1)
    volume_inv = 1.0 / volume
    reciprocal[..., 0] = volume_inv * torch.linalg.cross(primitive[..., 1], primitive[..., 2])
    reciprocal[..., 1] = volume_inv * torch.linalg.cross(primitive[..., 2], primitive[..., 0])
    reciprocal[..., 2] = volume_inv * torch.linalg.cross(primitive[..., 0], primitive[..., 1])
    return reciprocal
