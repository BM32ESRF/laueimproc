#!/usr/bin/env python3

"""Simulation of the bragg diffraction."""

import torch

PLANK_H = 6.63e-34
CELERITY_C = 3.00e8


def reciprocal_hkl_to_energy(reciprocal: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """Alias to ``laueimproc.diffraction.bragg.reciprocal_hkl_to_uq_energy``."""
    _, energy = reciprocal_hkl_to_uq_energy(reciprocal, hkl, _return_uq=False)
    return energy


def reciprocal_hkl_to_uq(reciprocal: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """Alias to ``laueimproc.diffraction.bragg.reciprocal_hkl_to_uq_energy``."""
    u_q, _ = reciprocal_hkl_to_uq_energy(reciprocal, hkl, _return_energy=False)
    return u_q


def reciprocal_hkl_to_uq_energy(
    reciprocal: torch.Tensor, hkl: torch.Tensor,
    *, _return_uq: bool = True, _return_energy: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Thanks to the bragg relation, compute the energy of each diffracted ray.

    The incidient ray \(u_i\) is assumed to be \(\mathbf{L_1}\),
    ie \([0, 0, 1]\) in the lab base \(\mathcal{B^l}\).

    Parameters
    ----------
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) of shape (*r, 3, 3) in the lab base \(\mathcal{B^l}\).
    hkl : torch.Tensor
        The h, k, l indices of shape (*n, 3) we want to mesure.

    Returns
    -------
    uq : torch.Tensor
        All the unitary diffracting plane normal vector of shape (*r, *n, 3).
        The vectors are expressed in the same base as the reciprocal space.
    enery : torch.Tensor
        The energy of each ray in J as a tensor of shape (*r, *n).
        \(\begin{cases}
            E = \frac{h.c}{\lambda} \\
            \lambda = 2.d.\sin(\theta) \\
            \sin(\theta) = |<u_i, u_q>| \\
        \end{cases}\)

    Examples
    --------
    >>> import torch
    >>> from laueimproc.diffraction.bragg import reciprocal_hkl_to_uq_energy
    >>> reciprocal = torch.torch.tensor([[ 1.6667e+09,  0.0000e+00,  0.0000e+00],
    ...                                  [ 9.6225e+08,  3.0387e+09, -0.0000e+00],
    ...                                  [-6.8044e+08, -2.1488e+09,  8.1653e+08]])
    >>> hkl = torch.tensor([1, 2, -1])
    >>> u_q, energy = reciprocal_hkl_to_uq_energy(reciprocal, hkl)
    >>> u_q
    tensor([ 0.1798,  0.7595, -0.6252])
    >>> 6.24e18 * energy  # convertion J -> eV
    tensor(9200.6816)
    >>>
    """
    assert isinstance(reciprocal, torch.Tensor), reciprocal.__class__.__name__
    assert reciprocal.shape[-2:] == (3, 3), reciprocal.shape
    assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
    assert hkl.shape[-1:] == (3,), hkl.shape
    assert not hkl.dtype.is_complex and not hkl.dtype.is_floating_point, hkl.dtype

    *batch_r, _, _ = reciprocal.shape
    *batch_n, _ = hkl.shape

    hkl = hkl[*((None,)*len(batch_r)), ..., None]  # (*r, *n, 3, 1)
    reciprocal = reciprocal[..., *((None,)*len(batch_n)), :, :]  # (*r, *n, 3, 3)
    u_q = reciprocal @ hkl.to(reciprocal.device, reciprocal.dtype)  # (*r, *n, 3, 1)
    inv_d_square = u_q.mT @ u_q  # (*r, *n, 1, 1)
    u_q = u_q.squeeze(-1)  # (*r, *n, 3)
    inv_d_square = inv_d_square.squeeze(-1)  # (*r, *n, 1)

    enery = None
    if _return_energy:
        ui_dot_uq = u_q[..., 2]  # (*r, *n)
        inv_d_sin_theta = inv_d_square.squeeze(-1) / ui_dot_uq
        enery = torch.abs((0.5 * PLANK_H * CELERITY_C) * inv_d_sin_theta)

    if _return_uq:
        u_q = u_q * torch.rsqrt(inv_d_square)

    return u_q, enery


def _select_all_hkl(max_hkl: int, dtype: torch.dtype) -> torch.Tensor:
    # create all candidates
    steps = (
        torch.arange(max_hkl+1, dtype=torch.int16),  # [h, -k, -l] = [-h, k, l]
        torch.arange(-max_hkl, max_hkl+1, dtype=torch.int16),
        torch.arange(-max_hkl, max_hkl+1, dtype=torch.int16),
    )
    steps = torch.meshgrid(*steps, indexing="ij")
    steps = torch.cat([s.reshape(-1, 1) for s in steps], dim=1)  # (n, 3)
    # reject bad candidates
    cond = torch.sum(torch.abs(steps), dim=1) <= max_hkl
    steps = steps[cond, :]
    cond = torch.gcd(torch.gcd(steps[:, 0], steps[:, 1]), steps[:, 2]) == 1  # remove harmonics
    steps = steps[cond, :]
    # convertion cast
    steps = steps.to(dtype)
    return steps
