#!/usr/bin/env python3

"""Help to select the hkl indices."""

import functools
import numbers

import torch

from .bragg import PLANK_H, CELERITY_C
from .lattice import lattice_to_primitive
from .reciprocal import primitive_to_reciprocal


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
    max_hkl: numbers.Integral = 18,
    e_max: numbers.Real = torch.inf,
    keep_harmonics: bool = True,
) -> torch.Tensor:
    r"""Reject the hkl sistematicaly out of energy band.

    Parameters
    ----------
    lattice : torch.Tensor, optional
        If provided, the lattices parameters \([a, b, c, \alpha, \beta, \gamma]]\) of shape (..., 6)
        are used to simulate the hkl energy for a large number of rotation.
    max_hkl : int
        The maximum absolute hkl sum such as |h| + |k| + |l| <= max_hkl.
    e_max : float, optional
        Indicies that systematicaly have an energy strictely greater than `e_max` in J are rejected.
    keep_harmonics : boolean, default=True
        If False, delete the multiple hkl indicies of the other.
        In other words, keep only the highest energy armonic.

    Returns
    -------
    hkl : torch.Tensor
        The int16 hkl of shape (n, 3).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.hkl import select_hkl
    >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
    >>> select_hkl(lattice)
    tensor([[  0, -18,   0],
            [  0, -17,  -1],
            [  0, -17,   0],
            ...,
            [ 17,   0,   1],
            [ 17,   1,   0],
            [ 18,   0,   0]], dtype=torch.int16)
    >>> len(_)
    4578
    >>> select_hkl(lattice, keep_harmonics=False)
    tensor([[  0, -17,  -1],
            [  0, -17,   1],
            [  0, -16,  -1],
            ...,
            [ 17,   0,  -1],
            [ 17,   0,   1],
            [ 17,   1,   0]], dtype=torch.int16)
    >>> len(_)
    3661
    >>> select_hkl(lattice, e_max=20e3 * 1.60e-19)  # 20 keV
    tensor([[  0, -11,  -3],
            [  0, -11,  -2],
            [  0, -11,  -1],
            ...,
            [ 11,   3,   0],
            [ 11,   3,   1],
            [ 11,   3,   2]], dtype=torch.int16)
    >>> len(_)
    3399
    >>>
    """
    assert isinstance(max_hkl, numbers.Integral), max_hkl.__class__.__name__
    assert 0 < max_hkl <= torch.iinfo(torch.int16).max, max_hkl
    assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
    assert isinstance(keep_harmonics, bool), keep_harmonics.__class__.__name__
    assert isinstance(lattice, torch.Tensor | None), lattice.__class__.__name__
    assert lattice is None or lattice.shape[-1:] == (6,), lattice.shape

    hkl = _select_all_hkl(int(max_hkl))  # (n, 3)
    if lattice is not None:
        hkl = hkl.to(lattice.device)

    if lattice is not None and e_max < torch.inf:
        reciprocal = primitive_to_reciprocal(lattice_to_primitive(lattice))  # (..., 3, 3)
        u_q = reciprocal[..., None, :, :] @ hkl[..., None].to(reciprocal.device, reciprocal.dtype)
        inv_d_square = u_q.mT @ u_q  # (..., n, 1, 1)
        energy = 0.5 * CELERITY_C * PLANK_H * torch.sqrt(inv_d_square[..., :, 0, 0])  # (..., n)
        energy = energy.reshape(-1, len(hkl))
        # import ipdb; ipdb.set_trace()
        hkl = hkl[torch.any(energy <= e_max, dim=0)]  # energy corresponds to the energy max
    else:
        energy = None

    if not keep_harmonics:
        hkl = hkl[torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2]) == 1]
        # method to select only the first diffracting harmonic, if e_min > 0...
        # family = hkl // torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2]).unsqueeze(-1)
        # family = family.tolist()
        # family_dict = {}  # to each family, associate the hkls
        # hkl = hkl.tolist()
        # for family_, hkl_ in zip(family, hkl):
        #     family_ = tuple(family_)
        #     family_dict[family_] = family_dict.get(family_, [])
        #     family_dict[family_].append(hkl_)
        # family_dict = {  # to each family, associate the hkl to keep
        #     family_: min(hkl_list, key=lambda hkl_: sum(map(abs, hkl_)))
        #     for family_, hkl_list in family_dict.items()
        # }
        # hkl = [
        #     hkl_ for hkl_, family_ in zip(hkl, family)
        #     if family_dict[tuple(family_)] == hkl_
        # ]
        # hkl = torch.tensor(hkl, device=lattice.device, dtype=torch.int16)

    return hkl
