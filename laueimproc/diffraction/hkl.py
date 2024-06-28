#!/usr/bin/env python3

"""Help to select the hkl indices."""

import functools
import numbers

import torch

from .bragg import hkl_reciprocal_to_energy
from .lattice import lattice_to_primitive
from .reciprocal import primitive_to_reciprocal
from .rotation import angle_to_rot, rotate_cristal


@functools.lru_cache(maxsize=8)
def _select_all_hkl(max_hkl: int) -> torch.Tensor:
    """Return a scan of all hkl candidates.

    Parameters
    ----------
    max_hkl : int
        The maximum absolute hkl sum such as |h| + |k| + |l| <= max_hkl.

    Returns
    -------
    hkl : torch.Tensor
        The int16 hkl of shape (n, 3).
    """
    # create all candidates
    steps = (
        torch.arange(max_hkl+1, dtype=torch.int16),  # [h, -k, -l] = [-h, k, l]
        torch.arange(-max_hkl, max_hkl+1, dtype=torch.int16),
        torch.arange(-max_hkl, max_hkl+1, dtype=torch.int16),
    )
    steps = torch.meshgrid(*steps, indexing="ij")
    steps = torch.cat([s.reshape(-1, 1) for s in steps], dim=1)  # (n, 3)

    # reject bad candidates
    cond = torch.sum(torch.abs(steps), dim=1) <= max_hkl  # |h| + |k| + |l| <= max_hkl
    cond[2*max_hkl*(1 + max_hkl)] = False  # remove (h, k, l) = (0, 0, 0)
    steps = steps[cond, :]

    return steps


def select_hkl(
    lattice: torch.Tensor | None = None,
    *,
    max_hkl: numbers.Integral = 10,
    e_min: numbers.Real = 0.0,
    e_max: numbers.Real = torch.inf,
    keep_harmonics: bool = True,
) -> torch.Tensor:
    r"""Simulate the hkl energy for different rotations, reject out of band indices.

    Parameters
    ----------
    lattice : torch.Tensor, optional
        If provided, the lattices parameters \([a, b, c, \alpha, \beta, \gamma]]\) of shape (..., 6)
        are used to simulate the hkl energy for a large number of rotation.
    max_hkl : int
        The maximum absolute hkl sum such as |h| + |k| + |l| <= max_hkl.
    e_min : float, optional
        Indicies that systematicaly have an energy strictely less than `e_min` in J are rejected.
    e_max : float, optional
        Indicies that systematicaly have an energy strictely greater than `e_max` in J are rejected.
    keep_harmonics : boolean, default=True
        If False, delete the multiple hkl indicies of the other.
        In other words, keep only the highest energy armonic.

    Returns
    -------
    hkl : torch.Tensor
        The int16 hkl of shape (n, 3).
    """
    assert isinstance(max_hkl, numbers.Integral), max_hkl.__class__.__name__
    assert 0 < max_hkl <= torch.iinfo(torch.int16).max, max_hkl
    assert isinstance(e_min, numbers.Real), e_min.__class__.__name__
    assert e_min >= 0.0, e_min
    assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
    assert e_max > e_min, (e_min, e_max)
    assert isinstance(keep_harmonics, bool), keep_harmonics.__class__.__name__
    assert isinstance(lattice, torch.Tensor | None), lattice.__class__.__name__
    assert lattice is None or lattice.shape[-1:] == (6,), lattice.shape

    hkl = _select_all_hkl(int(max_hkl))
    if lattice is not None:
        hkl = hkl.to(lattice.device)

    if lattice is not None and (e_min > 0.0 or e_max < torch.inf):
        reciprocal = primitive_to_reciprocal(lattice_to_primitive(lattice))
        angles = torch.linspace(
            -torch.pi/2, torch.pi/2, 12, device=lattice.device, dtype=lattice.dtype
        )
        reciprocal = rotate_cristal(reciprocal, angle_to_rot(angles, angles, angles))
        energy = hkl_reciprocal_to_energy(hkl, reciprocal).reshape(len(hkl), -1)
        hkl = hkl[torch.any(energy >= e_min, dim=1) & torch.any(energy <= e_max, dim=1)]
    else:
        energy = False

    if not keep_harmonics:
        if energy is None:
            hkl = hkl[torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2]) == 1]
        else:
            family = hkl // torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2]).unsqueeze(-1)
            family = family.tolist()
            family_dict = {}  # to each family, associate the hkls
            hkl = hkl.tolist()
            for family_, hkl_ in zip(family, hkl):
                family_ = tuple(family_)
                family_dict[family_] = family_dict.get(family_, [])
                family_dict[family_].append(hkl_)
            family_dict = {  # to each family, associate the hkl to keep
                family_: min(hkl_list, key=lambda hkl_: sum(map(abs, hkl_)))
                for family_, hkl_list in family_dict.items()
            }
            hkl = [
                hkl_ for hkl_, family_ in zip(hkl, family)
                if family_dict[tuple(family_)] == hkl_
            ]
            hkl = torch.tensor(hkl, device=lattice.device, dtype=torch.int16)

    return hkl
