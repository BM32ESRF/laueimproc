#!/usr/bin/env python3

"""Help to select the hkl indices."""

import functools
import math
import numbers

import torch

from .bragg import PLANK_H, CELERITY_C


@functools.lru_cache(maxsize=8)
def _select_all_hkl(hkl_max: int, device: torch.device = "cpu") -> torch.Tensor:
    """Return a scan of all hkl candidates.

    Parameters
    ----------
    hkl_max : int
        The maximum absolute hkl sum such as |h| + |k| + |l| <= hkl_max.
    device : torch.device, default="cpu"
        The device in which the tensors will be created.

    Returns
    -------
    hkl : torch.Tensor
        The int16 hkl of shape (n, 3).
    """
    # create all candidates
    steps = (
        torch.arange(hkl_max+1, dtype=torch.int16, device=device),  # [h, -k, -l] = [-h, k, l]
        torch.arange(-hkl_max, hkl_max+1, dtype=torch.int16, device=device),
        torch.arange(-hkl_max, hkl_max+1, dtype=torch.int16, device=device),
    )
    steps = torch.meshgrid(*steps, indexing="ij")
    steps = torch.cat([s.reshape(-1, 1) for s in steps], dim=1)  # (n, 3)

    # reject bad candidates
    cond = torch.sum(torch.abs(steps), dim=1) <= hkl_max  # |h| + |k| + |l| <= hkl_max
    cond[2*hkl_max*(1 + hkl_max)] = False  # remove (h, k, l) = (0, 0, 0)
    steps = steps[cond, :]

    return steps


def select_hkl(
    reciprocal: torch.Tensor | None = None,
    *,
    hkl_max: numbers.Integral | None = None,
    e_max: numbers.Real = torch.inf,
    keep_harmonics: bool = True,
) -> torch.Tensor:
    r"""Reject the hkl sistematicaly out of energy band.

    Parameters
    ----------
    reciprocal : torch.Tensor, optional
        Matrix \(\mathbf{B}\) in any orthonormal base.
    hkl_max : int, optional
        The maximum absolute hkl sum such as |h| + |k| + |l| <= hkl_max.
        If it is not provided, it is automaticaly find with the max energy.
    e_max : float, optional
        Indicies that systematicaly have an energy strictely greater than `e_max` in J are rejected.
    keep_harmonics : boolean, default=True
        If False, delete the multiple hkl indices of the other.
        In other words, keep only the highest energy armonic.

    Returns
    -------
    hkl : torch.Tensor
        The int16 hkl of shape (n, 3).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.hkl import select_hkl
    >>> reciprocal = torch.tensor([[2.7778e9,        0,        0],
    ...                            [1.2142e2, 2.7778e9,        0],
    ...                            [1.2142e2, 1.2142e2, 2.7778e9]])
    >>> select_hkl(hkl_max=18)
    tensor([[  0, -18,   0],
            [  0, -17,  -1],
            [  0, -17,   0],
            ...,
            [ 17,   0,   1],
            [ 17,   1,   0],
            [ 18,   0,   0]], dtype=torch.int16)
    >>> len(_)
    4578
    >>> select_hkl(hkl_max=18, keep_harmonics=False)
    tensor([[  0, -17,  -1],
            [  0, -17,   1],
            [  0, -16,  -1],
            ...,
            [ 17,   0,  -1],
            [ 17,   0,   1],
            [ 17,   1,   0]], dtype=torch.int16)
    >>> len(_)
    3661
    >>> select_hkl(reciprocal, e_max=20e3 * 1.60e-19)  # 20 keV
    tensor([[  0, -11,  -3],
            [  0, -11,  -2],
            [  0, -11,  -1],
            ...,
            [ 11,   3,   0],
            [ 11,   3,   1],
            [ 11,   3,   2]], dtype=torch.int16)
    >>> len(_)
    3519
    >>> select_hkl(reciprocal, e_max=20e3 * 1.60e-19, keep_harmonics=False)  # 20 keV
    tensor([[  0, -11,  -3],
            [  0, -11,  -2],
            [  0, -11,  -1],
            ...,
            [ 11,   3,   0],
            [ 11,   3,   1],
            [ 11,   3,   2]], dtype=torch.int16)
    >>> len(_)
    2889
    >>>
    """
    assert isinstance(hkl_max, numbers.Integral | None), hkl_max.__class__.__name__
    assert hkl_max is None or 0 < hkl_max <= torch.iinfo(torch.int16).max, hkl_max
    assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
    assert isinstance(keep_harmonics, bool), keep_harmonics.__class__.__name__
    assert isinstance(reciprocal, torch.Tensor | None), reciprocal.__class__.__name__
    assert reciprocal is None or reciprocal.shape[-2:] == (3, 3), reciprocal.shape

    if hkl_max is None:
        assert reciprocal is not None and e_max < torch.inf, \
            "you have to provide 'hkl_max' or 'reciprocal' and 'e_max'"
        q_norm = math.sqrt(float(torch.min(torch.sum(reciprocal * reciprocal, dim=-2))))
        hkl_max = math.ceil(3.47 * e_max / (CELERITY_C * PLANK_H * q_norm))  # |q| < 2*e_max / (H*C)

    hkl = _select_all_hkl(  # (n, 3)
        int(hkl_max), device=(torch.device("cpu") if reciprocal is None else reciprocal.device)
    )

    if reciprocal is not None and e_max < torch.inf:
        u_q = reciprocal[..., None, :, :] @ hkl[..., None].to(reciprocal.device, reciprocal.dtype)
        inv_d_square = u_q.mT @ u_q  # (..., n, 1, 1)
        energy = 0.5 * CELERITY_C * PLANK_H * torch.sqrt(inv_d_square[..., :, 0, 0])  # (..., n)
        energy = energy.reshape(-1, len(hkl))  # (:, n)
        hkl = hkl[torch.any(energy <= e_max, dim=0)]  # energy corresponds to the energy max
    else:
        energy = None

    if not keep_harmonics:
        hkl = hkl[torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2]) == 1]
        # method to select only the first diffracting harmonic, if e_min > 0 ...
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
