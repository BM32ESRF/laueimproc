#!/usr/bin/env python3

r"""Enables communication between primitive \(\mathbf{A}\) and reciprocal space \(\mathbf{B}\)."""

import torch


def primitive_to_reciprocal(primitive: torch.Tensor) -> torch.Tensor:
    r"""Convert the primitive vectors into the reciprocal base vectors.

    Bijection of ``laueimproc.geometry.reciprocal.reciprocal_to_primitive``.

    .. image:: ../../../build/media/IMGPrimitiveReciprocal.avif

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
    >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal
    >>> primitive = torch.tensor([[ 6.0000e-10, -1.9000e-10, -6.5567e-17],
    ...                           [ 0.0000e+00,  3.2909e-10,  8.6603e-10],
    ...                           [ 0.0000e+00,  0.0000e+00,  1.2247e-09]])
    >>> primitive_to_reciprocal(primitive)
    tensor([[ 1.6667e+09,  0.0000e+00,  0.0000e+00],
            [ 9.6225e+08,  3.0387e+09, -0.0000e+00],
            [-6.8044e+08, -2.1488e+09,  8.1653e+08]])
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


def reciprocal_to_primitive(reciprocal: torch.Tensor) -> torch.Tensor:
    r"""Convert the reciprocal vectors into the primitive base vectors.

    Bijection of ``laueimproc.geometry.reciprocal.primitive_to_reciprocal``.

    .. image:: ../../../build/media/IMGPrimitiveReciprocal.avif

    Parameters
    ----------
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) in any orthonormal base.

    Returns
    -------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) in the same orthonormal base.
    """
    return primitive_to_reciprocal(reciprocal)  # inv(f) = f
