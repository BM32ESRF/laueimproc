#!/usr/bin/env python3

"""Find the crystal symmetries and equivalence."""

import itertools
import numbers

import torch

from .bragg import hkl_reciprocal_to_uq


def find_symmetric_rotations(crystal: torch.Tensor, tol: numbers.Real = 0.05) -> torch.Tensor:
    r"""Search all the rotation that keep the crystal identical.

    Parameters
    ----------
    crystal : torch.Tensor
        The primitive \((\mathbf{A})\) or reciprocal \((\mathbf{B})\)
        in any orthonormal base, of shape (3, 3).
    tol : float, default=0.05
        The tolerency in percent to consider that to matrices are the same.

    Returns
    -------
    rot : torch.Tensor
        All the rotation matrices leaving the crystal equivalent.
        The shape is (n, 3, 3). It contains the identity matrix.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.symmetry import find_symmetric_rotations
    >>> primitive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> find_symmetric_rotations(primitive).shape
    torch.Size([24, 3, 3])
    >>> primitive = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> find_symmetric_rotations(primitive).shape
    torch.Size([8, 3, 3])
    >>> primitive = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    >>> find_symmetric_rotations(primitive).shape
    torch.Size([4, 3, 3])
    >>> primitive = torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.866, 0.0], [0.0, 0.0, 1.0]])
    >>> find_symmetric_rotations(primitive).shape
    torch.Size([4, 3, 3])
    >>> primitive = torch.tensor([[1.0, 0.866, 0.866], [0.0, 0.5, 0.232], [0.0, 0.0, 0.443]])
    >>> find_symmetric_rotations(primitive).shape  # a=b=c := 1, alpha=beta=gamma := pi/6
    torch.Size([6, 3, 3])
    >>>
    """
    assert isinstance(crystal, torch.Tensor), crystal.__class__.__name__
    assert crystal.shape == (3, 3), crystal.__class__.__name__
    assert isinstance(tol, numbers.Real), tol.__class__.__name__
    assert 0.0 <= tol < 1.0, tol

    # test all permutations
    all_prim = crystal[:, list(itertools.permutations([0, 1, 2]))].movedim(1, 0)  # (6, 3, 3)
    norm = torch.sum(crystal * crystal, dim=0)  # (3,)
    all_prim = all_prim[  # (n, 3, 3)
        torch.all(  # allow permutation if the norm doesn't change
            torch.abs((torch.sum(all_prim * all_prim, dim=1) - norm) / norm) < tol,
            dim=1,
        )
    ]

    # set all symmetries
    sym = torch.tensor(list(  # (8, 3)
        itertools.product([1, -1], [1, -1], [1, -1])
    ), dtype=all_prim.dtype, device=all_prim.device)
    all_prim = all_prim[None, :, :, :] * sym[:, None, None, :]  # (n, 8, 3, 3)
    all_prim = all_prim.reshape(-1, 3, 3)

    # find all equivalent transition matrices
    rot = all_prim @ torch.linalg.inv(crystal)  # L' = R.L

    # reject non rotation matrix
    det = torch.linalg.det(rot)
    rot = rot[torch.abs(det - 1.0) < tol]  # det of rot matrix equal 1
    identity = rot @ rot.mT  # orthogonal
    rot = rot[
        torch.all(
            torch.abs(identity - torch.eye(3, dtype=identity.dtype, device=identity.device)) < tol,
            dim=(1, 2)
        )
    ]
    return rot


def hkl_family(
    hkl: torch.Tensor, reciprocal: torch.Tensor, tol: numbers.Real = 0.05
) -> torch.Tensor:
    r"""Find all the equivalent hkl by symetry.

    Parameters
    ----------
    hkl : torch.Tensor
        The real int hkl indices, of shape (n, 3).
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) in any orthonormal base, of shape (n, 3).
        It is used to find the symmetries.
    tol : float, default=0.05
        The tolerency in percent to rounded the hkl and finding the symmetries.

    Returns
    -------
    expanded_hkl : torch.Tensor
        All the int32 equivalent hkl, of shape (n, s, 3), with s the number of symmetries.

    Examples
    --------
    >>> from pprint import pprint
    >>> import torch
    >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal
    >>> from laueimproc.geometry.symmetry import hkl_family
    >>> hkl = torch.tensor([[0, 0, 1], [2, 2, 2], [0, 1, 2]])
    >>> primitive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> family = hkl_family(hkl, primitive_to_reciprocal(primitive))
    >>> pprint([set(map(tuple, f)) for f in family.tolist()])
    [{(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, -1), (0, 0, 1)},
     {(-1, -1, -1),
      (-1, -1, 1),
      (-1, 1, -1),
      (-1, 1, 1),
      (1, -1, -1),
      (1, -1, 1),
      (1, 1, -1),
      (1, 1, 1)},
     {(-2, -1, 0),
      (-2, 0, -1),
      (-2, 0, 1),
      (-2, 1, 0),
      (-1, -2, 0),
      (-1, 0, -2),
      (-1, 0, 2),
      (-1, 2, 0),
      (0, -2, -1),
      (0, -2, 1),
      (0, -1, -2),
      (0, -1, 2),
      (0, 1, -2),
      (0, 1, 2),
      (0, 2, -1),
      (0, 2, 1),
      (1, -2, 0),
      (1, 0, -2),
      (1, 0, 2),
      (1, 2, 0),
      (2, -1, 0),
      (2, 0, -1),
      (2, 0, 1),
      (2, 1, 0)}]
    >>> primitive = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> family = hkl_family(hkl, primitive_to_reciprocal(primitive))
    >>> pprint([set(map(tuple, f)) for f in family.tolist()])
    [{(0, 0, -1), (0, -1, 0), (0, 0, 1), (0, 1, 0)},
     {(-1, -1, -1),
      (-1, -1, 1),
      (-1, 1, -1),
      (-1, 1, 1),
      (1, -1, -1),
      (1, -1, 1),
      (1, 1, -1),
      (1, 1, 1)},
     {(0, -2, -1),
      (0, -2, 1),
      (0, -1, -2),
      (0, -1, 2),
      (0, 1, -2),
      (0, 1, 2),
      (0, 2, -1),
      (0, 2, 1)}]
    >>>
    """
    assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
    assert hkl.ndim == 2 and hkl.shape[1] == 3, hkl.shape
    assert isinstance(reciprocal, torch.Tensor), reciprocal.__class__.__name__
    assert reciprocal.shape == (3, 3), reciprocal.shape
    assert isinstance(tol, numbers.Real), tol.__class__.__name__
    assert 4.6566e-10 < tol < 1.0, tol  # strict because divided by tol and cast in int32

    # from hkl and reciprocal to uq
    uq_ref = hkl_reciprocal_to_uq(hkl, reciprocal)  # (n, 3)

    # from uq and symmetries to uq_family
    sym_rot = find_symmetric_rotations(reciprocal, tol=tol)  # (r, 3, 3)
    uq_family = sym_rot[None, :, :] @ uq_ref[:, None, :, None]  # (n, r, 3, 1), all equivalent uq

    # from uq_family and reciprocal to float hkl
    # we have: lambda.uq = h.e1* + k.e2* + l.e2*
    # => <uq, ei*> = h.<ei*, e1*> + k.<ei*, e2*> + l.<ei*, e3*>
    uq_proj = reciprocal.mT @ uq_family  # (n, r, 3, 1), <uq, ei*>
    scal_matrix = reciprocal.mT @ reciprocal  # (3, 3), <ei*, ej*>
    hkl = torch.linalg.inv(scal_matrix) @ uq_proj  # (n, r, 3, 1), uq_proj = scal_matrix @ hkl

    # from float hkl to int hkl
    hkl = hkl.squeeze(3)  # (n, r, 3)
    hkl *= torch.rsqrt(torch.sum(hkl * hkl, dim=2, keepdim=True)) / tol  # |hkl| = 1/tol
    hkl = torch.round(hkl).to(torch.int32)  # round big then make as small as possible
    hkl //= torch.gcd(torch.gcd(hkl[:, :, 0], hkl[:, :, 1]), hkl[:, :, 2]).unsqueeze(2)

    return hkl
