#!/usr/bin/env python3

r"""Link the lattice parameters and the primitive vectors \(\mathbf{A}\)."""

import torch


def lattice_to_primitive(lattice: torch.Tensor) -> torch.Tensor:
    r"""Convert the lattice parameters into primitive vectors.

    .. image:: ../../../build/media/IMGLatticeBc.avif
        :width: 256

    Parameters
    ----------
    lattice : torch.Tensor
        The array of lattice parameters of shape (..., 6).
        Values are \([a, b, c, \alpha, \beta, \gamma]]\).

    Returns
    -------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) of shape (..., 3, 3) in the base \(\mathcal{B^c}\).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.lattice import lattice_to_primitive
    >>> lattice = torch.tensor([6.0e-10, 3.8e-10, 15e-10, torch.pi/3, torch.pi/2, 2*torch.pi/3])
    >>> lattice_to_primitive(lattice)  # quartz lattice
    tensor([[ 6.0000e-10, -1.9000e-10, -6.5567e-17],
            [ 0.0000e+00,  3.2909e-10,  8.6603e-10],
            [ 0.0000e+00,  0.0000e+00,  1.2247e-09]])
    >>>
    """
    assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
    assert lattice.shape[-1:] == (6,), lattice.shape

    lat_a, lat_b, lat_c, alpha, beta, gamma = lattice.movedim(-1, 0)
    zero = torch.zeros_like(lat_a)
    cos_beta = torch.cos(beta)
    sin_gamma = torch.sin(gamma)
    cos_gamma = torch.cos(gamma)
    e3_c2 = (torch.cos(alpha) - cos_gamma*cos_beta) / sin_gamma

    return torch.cat([
        lat_a.unsqueeze(-1),  # e1_c1
        (lat_b * cos_gamma).unsqueeze(-1),  # e2_c1
        (lat_c * cos_beta).unsqueeze(-1),  # e3_c1
        zero.unsqueeze(-1),  # e1_c2
        (lat_b * sin_gamma).unsqueeze(-1),  # e2_c2
        (lat_c * e3_c2).unsqueeze(-1),  # e3_c2
        zero.unsqueeze(-1),  # e1_c3
        zero.unsqueeze(-1),  # e2_c3
        (lat_c * torch.sqrt(torch.sin(beta)**2 - e3_c2**2)).unsqueeze(-1),  # e3_c3
    ], dim=-1).reshape(*lattice.shape[:-1], 3, 3)


def primitive_to_lattice(primitive: torch.Tensor) -> torch.Tensor:
    r"""Convert the primitive vectors to the lattice parameters.

    .. image:: ../../../build/media/IMGLattice.avif
        :width: 256

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

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.lattice import primitive_to_lattice
    >>> primitive = torch.tensor([[ 6.0000e-10, -1.9000e-10, -6.5567e-17],
    ...                           [ 0.0000e+00,  3.2909e-10,  8.6603e-10],
    ...                           [ 0.0000e+00,  0.0000e+00,  1.2247e-09]])
    >>> primitive_to_lattice(primitive)  # quartz lattice
    tensor([6.0000e-10, 3.8000e-10, 1.5000e-09, 1.0472e+00, 1.5708e+00, 2.0944e+00])
    >>>
    """
    assert isinstance(primitive, torch.Tensor), primitive.__class__.__name__
    assert primitive.shape[-2:] == (3, 3), primitive.shape

    abc = torch.sqrt(torch.sum(primitive * primitive, dim=-2))  # (..., 3)
    e_123 = primitive / abc.unsqueeze(-2)  # (..., 3, 3)
    e_1, e_2, e_3 = e_123.movedim(-1, 0)
    alpha = torch.acos(torch.sum(e_2 * e_3, dim=-1, keepdim=True))  # (..., 1)
    beta = torch.acos(torch.sum(e_1 * e_3, dim=-1, keepdim=True))  # (..., 1)
    gamma = torch.acos(torch.sum(e_1 * e_2, dim=-1, keepdim=True))  # (..., 1)
    return torch.cat([abc, alpha, beta, gamma], dim=-1)
