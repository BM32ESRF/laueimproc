#!/usr/bin/env python3

"""Test the function of bragg diffraction."""

import torch

from laueimproc.diffraction.bragg import (
    hkl_reciprocal_to_energy, hkl_reciprocal_to_uq, uf_to_uq, uq_to_uf
)


RECIPROCAL = torch.eye(3)  # cubic in Bc
BATCH_RECIPROCAL = (10, 11, 12)
BATCHED_RECIPROCAL = RECIPROCAL[None, None, None, :, :].expand(*BATCH_RECIPROCAL, -1, -1).clone()
HKL = torch.tensor([1, 1, 1])
BATCH_HKL = (13, 14, 15)
BATCHED_HKL = HKL[None, None, None, :].expand(*BATCH_HKL, -1).clone()
UF = torch.tensor([1.0, 0.0, 0.0])
BATCH_UF = (16, 17, 18)
BATCHED_UF = UF[None, None, None, :].expand(*BATCH_UF, -1).clone()
UQ = torch.tensor([-0.7071, 0.0, 0.7071])
BATCH_UQ = (19, 20, 21)
BATCHED_UQ = UQ[None, None, None, :].expand(*BATCH_UQ, -1).clone()


def test_batch_hkl_reciprocal_to_energy():
    """Test batch dimension."""
    assert hkl_reciprocal_to_energy(
        torch.empty(0, 3, dtype=int), torch.empty(0, 3, 3)
    ).shape == (0, 0)
    assert hkl_reciprocal_to_energy(torch.empty(0, 3, dtype=int), RECIPROCAL).shape == (0,)
    assert hkl_reciprocal_to_energy(HKL, torch.empty(0, 3, 3)).shape == (0,)
    assert hkl_reciprocal_to_energy(HKL, RECIPROCAL).shape == ()
    assert hkl_reciprocal_to_energy(
        BATCHED_HKL, BATCHED_RECIPROCAL
    ).shape == BATCH_HKL + BATCH_RECIPROCAL


def test_batch_hkl_reciprocal_to_uq():
    """Test batch dimension."""
    assert hkl_reciprocal_to_uq(
        torch.empty(0, 3, dtype=int), torch.empty(0, 3, 3)
    ).shape == (0, 0, 3)
    assert hkl_reciprocal_to_uq(torch.empty(0, 3, dtype=int), RECIPROCAL).shape == (0, 3)
    assert hkl_reciprocal_to_uq(HKL, torch.empty(0, 3, 3)).shape == (0, 3)
    assert hkl_reciprocal_to_uq(HKL, RECIPROCAL).shape == (3,)
    assert hkl_reciprocal_to_uq(
        BATCHED_HKL, BATCHED_RECIPROCAL
    ).shape == (*BATCH_HKL, *BATCH_RECIPROCAL, 3)


def test_batch_uf_to_uq():
    """Test batch dimension."""
    assert uf_to_uq(torch.empty(0, 3)).shape == (0, 3)
    assert uf_to_uq(UF).shape == (3,)
    assert uf_to_uq(BATCHED_UF).shape == (*BATCH_UF, 3)


def test_batch_uq_to_uf():
    """Test batch dimension."""
    assert uq_to_uf(torch.empty(0, 3)).shape == (0, 3)
    assert uq_to_uf(UQ).shape == (3,)
    assert uq_to_uf(BATCHED_UQ).shape == (*BATCH_UQ, 3)


def test_bij_uf_to_uq_to_uf():
    """Test uf -> uq -> uf = uf."""
    u_f = torch.randn(1000, 3, dtype=torch.float64)
    u_f *= torch.rsqrt(torch.sum(u_f * u_f, dim=-1, keepdim=True))
    u_f_bis = uq_to_uf(uf_to_uq(u_f))
    assert torch.allclose(u_f, u_f_bis)


def test_bij_uq_to_uf_to_uq():
    """Test uq -> uf -> uq = uq."""
    u_q = torch.randn(1000, 3, dtype=torch.float64)
    u_q *= torch.rsqrt(torch.sum(u_q * u_q, dim=-1, keepdim=True))
    u_q_bis = uf_to_uq(uq_to_uf(u_q))
    assert torch.allclose(  # test colinear
        torch.linalg.cross(u_q, u_q_bis), torch.tensor(0.0, dtype=torch.float64)
    )
    assert torch.allclose(  # test same norm
        torch.linalg.vector_norm(u_q, dim=1), torch.linalg.vector_norm(u_q_bis, dim=1)
    )


def test_jac_hkl_reciprocal_to_energy():
    """Test compute jacobian."""
    assert torch.func.jacrev(hkl_reciprocal_to_energy, 1)(HKL, RECIPROCAL).shape == (3, 3)


def test_jac_hkl_reciprocal_to_uq():
    """Test compute jacobian."""
    assert torch.func.jacrev(hkl_reciprocal_to_uq, 1)(HKL, RECIPROCAL).shape == (3, 3, 3)


def test_normalization_hkl_reciprocal_to_uq():
    """Test norm is 1."""
    reciprocal = torch.eye(3) + torch.randn(1000, 3, 3) / 3
    hkl = torch.randint(-6, 7, (1000, 3))
    hkl = hkl[hkl.prod(dim=-1) != 0]
    u_q = hkl_reciprocal_to_uq(hkl, reciprocal)
    assert torch.allclose(torch.linalg.vector_norm(u_q, dim=-1), torch.tensor(1.0))


def test_normalization_uf_to_uq():
    """Test norm is 1."""
    u_f = torch.randn(1000, 3)
    u_f *= torch.rsqrt(torch.sum(u_f * u_f, dim=-1, keepdim=True))
    u_q = uf_to_uq(u_f)
    assert torch.allclose(torch.linalg.vector_norm(u_q, dim=-1), torch.tensor(1.0))


def test_normalization_uq_to_uf():
    """Test norm is 1."""
    u_q = torch.randn(1000, 3)
    u_q *= torch.rsqrt(torch.sum(u_q * u_q, dim=-1, keepdim=True))
    u_f = uq_to_uf(u_q)
    assert torch.allclose(torch.linalg.vector_norm(u_f, dim=-1), torch.tensor(1.0))


def test_sign_uq():
    """Test uf is not sensitive to the uq orientation."""
    u_q = torch.randn(1000, 3)
    u_q *= torch.rsqrt(torch.sum(u_q * u_q, dim=-1, keepdim=True))
    u_f_pos = uq_to_uf(u_q)
    u_f_neg = uq_to_uf(-u_q)
    assert torch.allclose(u_f_neg, u_f_pos)
