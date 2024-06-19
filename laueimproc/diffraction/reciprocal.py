#!/usr/bin/env python3

"""Functions to compute reciprocical lattice."""

import torch


def lattice_to_primitive(lattice: torch.Tensor) -> torch.Tensor:
    r"""Convert the lattice parameters into primitive vectors.

    Parameters
    ----------
    lattice : torch.Tensor
        The array of lattice parameters of shape (..., 6).
        Values are \([a, b, c, \alpha, \beta, \gamma]]\).

    Returns
    -------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) of shape (..., 3, 3) in the base \(\mathcal{B^c}\).
    """
    assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
    assert lattice.shape[-1:] == (6,), lattice.shape

    lat_a, lat_b, lat_c, alpha, beta, gamma = lattice.movedim(-1, 0)

    zero = torch.zeros_like(lat_a)
    e_1 = torch.cat([
        (lat_a * torch.sin(gamma)).unsqueeze(-1),
        (lat_a * torch.cos(gamma)).unsqueeze(-1),
        zero.unsqueeze(-1),
    ], dim=-1)
    e_2 = torch.cat([
        zero.unsqueeze(-1),
        (lat_b).unsqueeze(-1),
        zero.unsqueeze(-1),
    ], dim=-1)
    e_3 = torch.cat([
        (lat_c * torch.cos(beta)).unsqueeze(-1),
        (lat_c * torch.cos(alpha)).unsqueeze(-1),
        (lat_c * torch.sqrt(torch.sin(alpha)**2 + torch.sin(beta)**2 - 1)).unsqueeze(-1),
    ], dim=-1)
    primitive = torch.cat([e_1.unsqueeze(-1), e_2.unsqueeze(-1), e_3.unsqueeze(-1)], dim=-1)
    return primitive


def primitive_to_lattice(primitive: torch.Tensor) -> torch.Tensor:
    r"""Convert the primitive vectors to the lattice parameters.

    Parameters
    ----------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) in any orthonormal base.

    Returns
    -------
    lattice : torch.Tensor
        The array of lattice parameters of shape (..., 6).
        Values are \([a, b, c, \alpha, \beta, \gamma]]\).

    Notes
    -----
    We have `primitive_to_lattice(lattice_to_primitive(lattice)) == lattice`,
    but it is not the case for the composition inverse because the numerical
    value of \(\mathbf{A}\) is base dependent.
    """
    assert isinstance(primitive, torch.Tensor), primitive.__class__.__name__
    assert primitive.shape[-2:] == (3, 3), primitive.shape

    raise NotImplementedError


def primitive_to_reciprocal(primitive: torch.Tensor) -> torch.Tensor:
    r"""Convert the primitive vectors into the reciprocal base vectors.

    Based on https://fr.wikipedia.org/wiki/R%C3%A9seau_r%C3%A9ciproque.

    Parameters
    ----------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) in any orthonormal base.

    Returns
    -------
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) in the same orthonormal base.
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
