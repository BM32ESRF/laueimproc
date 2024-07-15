#!/usr/bin/env python3

"""Find the cristal symmetries and equivalence."""

import itertools

import torch


def find_symmetric_rotations(primitive: torch.Tensor) -> torch.Tensor:
    r"""Search all the rotation that keep the cristal identical.

    Parameters
    ----------
    primitive : torch.Tensor
        Matrix \(\mathbf{A}\) in any orthonormal base of shape (3, 3).

    Returns
    -------
    rot : torch.Tensor
        All the rotation matrices leaving the cristal equivalent.
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
    assert isinstance(primitive, torch.Tensor), primitive.__class__.__name__
    assert primitive.shape == (3, 3), primitive.__class__.__name__

    # define constants
    tol = 0.05  # 5% of tolerence

    # test all permutations
    all_prim = primitive[:, list(itertools.permutations([0, 1, 2]))].movedim(1, 0)  # (6, 3, 3)
    norm = torch.sum(primitive * primitive, dim=0)  # (3,)
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
    rot = all_prim @ torch.linalg.inv(primitive)  # L' = R.L

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


def reduce_hkl(hkl: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Keep a single representation of the equivalent hkl.

    Parameters
    ----------
    hkl : torch.Tensor
        The real int hkl indices, of shape (n, 3).
    rot : torch.Tensor
        The rotations matrices of symetries,
        comming from ``laueimproc.geometry.symmetry.find_symmetric_rotations``,
        of shape (r, 3, 3).

    Returns
    -------
    reduced_hkl : torch.Tensor
        The equivalent reduced hkl indices, of shape (n, 3).
    """
    assert isinstance(hkl, torch.Tensor), hkl.__class__.__name__
    assert hkl.ndim == 2 and hkl.shape[1] == 3, hkl.shape
    assert isinstance(rot, torch.Tensor), rot.__class__.__name__
    assert rot.ndim == 3 and rot.shape[1:] == (3, 3), rot.shape

    raise NotImplementedError("hmm, it it too complicated")

    # # remove harmonics
    # hkl = hkl // torch.gcd(torch.gcd(hkl[:, 0], hkl[:, 1]), hkl[:, 2])

    # # expand symmetries
    # hkl = hkl.to(rot.dtype)[:, None, :, None] @ rot[None, :, :, :]  # (n, r, 3, 1)
    # hkl = hkl.squeeze(3)  # (n, r, 3)

    # # select the better
    # def order(hkl1: list[int], hkl2: list[int]):
    #     """Return True if hkl1 > hkl2."""
    #     h_1, k_1, l_1 = hkl1
    #     h_2, k_2, l_2 = hkl2
    #     if (abs(h_1)+abs(k_1)+abs(l_1) > abs(h_1))

    # hkl_list = [min(hkl_s, key=order) in hkl.tolist()]
