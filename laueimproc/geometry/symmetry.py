#!/usr/bin/env python3

"""Find the crystal symmetries and equivalence."""

import itertools
import numbers

import torch


def find_symmetric_rotations(crystal: torch.Tensor, tol: numbers.Real = 0.05) -> torch.Tensor:
    r"""Search all the rotation that keep the crystal identical.

    Parameters
    ----------
    crystal : torch.Tensor
        The primitive \(\mathbf{A}\) or reciprocal \(\mathbf{B}\)
        in any orthonormal base, of shape (3, 3).
    tol : float, default=0.05
        The tolerency in percent to consider that to matrices are the same.

    Returns
    -------
    rot : torch.Tensor
        All the rotation matrices leaving the crystal equivalent.
        The shape is (n, 3, 3). It contains the identity matrix.

    Notes
    -----
    To understand this algorithm, it's simplest to think about \(\mathbf{A}\).

    Let \(f\) be the function that transforms primitive space into reciprocal space.
    This function is independent of the chosen orthonormal base,
    In particular, it is linear by rotation:

    \(
        \mathbf{B_{\mathcal{B^l}}
        = f\left( \mathbf{R} . \mathbf{A_{\mathcal{B^c}} \right)
        = \mathbf{R} . f\left( \mathbf{A_{\mathcal{B^c}} \right)
    \)

    The symmetries (which are rotation matrices) found on \(\mathbf{A}\)
    are therefore the same as the symmetries found on \(\mathbf{B}\).
    Thus, this function can be applied indefinitely to either \(\mathbf{A}\) or \(\mathbf{B}\).

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


def get_hkl_family_member(*args, **kwargs) -> torch.Tensor:
    """Select the canonical member of the hkl family.

    Parameters
    ----------
    hkl, reciprocal, tol
        Transmitted to ``laueimproc.geometry.symmetry.hkl.get_hkl_family_members``.

    Returns
    -------
    hkl : torch.Tensor
        The largest member of the hkl familly, according to an arbitrary partial order relation.
        The returned tensor is of shape (n, 3) and of type int16.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal
    >>> from laueimproc.geometry.symmetry import get_hkl_family_member
    >>> hkl = torch.tensor([[0, 0, 1], [2, 2, 2], [0, 1, 2]])
    >>> primitive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> get_hkl_family_member(hkl, primitive_to_reciprocal(primitive))
    tensor([[1, 0, 0],
            [1, 1, 1],
            [2, 1, 0]], dtype=torch.int16)
    >>> primitive = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> get_hkl_family_member(hkl, primitive_to_reciprocal(primitive))
    tensor([[0, 1, 0],
            [1, 1, 1],
            [0, 2, 1]], dtype=torch.int16)
    >>>
    """
    # the relation is reflexive: hkl <= hkl.
    # the relation is antisymmetric: hkl_1 <= hkl_2 and hkl_2 <= hkl_1 => hkl_1 = hkl_2.
    # the relation is transitive: hkl_1 <= hkl_2 and hkl_2 <= hkl_3 => hkl_1 <= hkl_3.
    all_hkl = get_hkl_family_members(*args, **kwargs)
    hkl = [max(map(tuple, members)) for members in all_hkl.tolist()]  # tuple order relation
    return torch.asarray(hkl, dtype=all_hkl.dtype, device=all_hkl.device)


def get_hkl_family_members(
    hkl: torch.Tensor, reciprocal: torch.Tensor, tol: numbers.Real = 0.05
) -> torch.Tensor:
    r"""Find all the members of the hkl family.

    A family consists of all irreductible hkl, invariant by crystal symmetry.

    Parameters
    ----------
    hkl : torch.Tensor
        The real int hkl indices, of shape (n, 3).
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) in any orthonormal base, of shape (3, 3).
        It is used to find the symmetries.
    tol : float, default=0.05
        The tolerency in percent to rounded the hkl and finding the symmetries.

    Returns
    -------
    all_hkl : torch.Tensor
        All the int32 equivalent hkl, of shape (n, s, 3), with s the number of symmetries.

    Examples
    --------
    >>> from pprint import pprint
    >>> import torch
    >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal
    >>> from laueimproc.geometry.symmetry import get_hkl_family_members
    >>> hkl = torch.tensor([[0, 0, 1], [2, 2, 2], [0, 1, 2]])
    >>> primitive = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> family = get_hkl_family_members(hkl, primitive_to_reciprocal(primitive))
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
    >>> family = get_hkl_family_members(hkl, primitive_to_reciprocal(primitive))
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
    assert 3.052e-5 < tol < 1.0, tol  # /tol and cast into int16

    hkl = hkl * round(1.0 / tol)  # expand to reduce after

    # from hkl and reciprocal to not unitary uq
    uq_ref = reciprocal[None, :, :] @ hkl[:, :, None].to(reciprocal.device, reciprocal.dtype)

    # from uq and symmetries to uq_family
    sym_rot = find_symmetric_rotations(reciprocal, tol=tol)  # (r, 3, 3)
    uq_family = sym_rot[None, :, :] @ uq_ref[:, None, :]  # (n, r, 3, 1), all equivalent uq

    # from uq_family and reciprocal to float hkl
    # we have: lambda.uq = h.e1* + k.e2* + l.e2*
    # => <uq, ei*> = h.<ei*, e1*> + k.<ei*, e2*> + l.<ei*, e3*>
    uq_proj = reciprocal.mT @ uq_family  # (n, r, 3, 1), <uq, ei*>
    scal_matrix = reciprocal.mT @ reciprocal  # (3, 3), <ei*, ej*>
    all_hkl = torch.linalg.inv(scal_matrix) @ uq_proj  # (n, r, 3, 1), uq_proj = scal_matrix @ hkl

    # from float hkl to int hkl
    all_hkl = all_hkl.squeeze(3)  # (n, r, 3)
    all_hkl_int = torch.round(all_hkl).to(torch.int16)  # round big then make as small as possible
    all_hkl_int //= (
        torch.gcd(torch.gcd(all_hkl_int[:, :, 0], all_hkl_int[:, :, 1]), all_hkl_int[:, :, 2])
        .unsqueeze(2)
    )
    return all_hkl_int

    # more accurate but slowler method to convert float hkl into int hkl
    # all_hkl *= torch.rsqrt(torch.sum(all_hkl * all_hkl, dim=2, keepdim=True))
    # hkl_max = 10
    # hkl_int = torch.round(   # (n, r, x, 3)
    #     all_hkl[:, :, None, :] * torch.sqrt(torch.arange(1, hkl_max**2 + 1)[None, None, :, None])
    # )
    # phi = torch.acos(
    #     torch.clamp(
    #         torch.sum(
    #             (
    #                 all_hkl[:, :, None, :]
    #                 * hkl_int
    #             ),
    #             dim=3,
    #         ) * torch.rsqrt(torch.sum(hkl_int * hkl_int, dim=3)),
    #         min=-1.0,
    #         max=1.0,
    #     )
    # )  # angle between all_hkl and hkl_int
    # best = torch.argmax((phi < tol).view(torch.uint8), dim=2, keepdim=True)
    # hkl_int = torch.take_along_dim(hkl_int, best[:, :, :, None], dim=2)  # (n, r, 3)
    # hkl_int = hkl_int.to(torch.int16)  # round big then make as small as possible
    # hkl_int //= (
    #     torch.gcd(torch.gcd(hkl_int[:, :, 0], hkl_int[:, :, 1]), hkl_int[:, :, 2])
    #     .unsqueeze(2)
    # )
    # return hkl_int
