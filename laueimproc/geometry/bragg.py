#!/usr/bin/env python3

"""Simulation of the bragg diffraction."""

import torch

PLANK_H = 6.63e-34
CELERITY_C = 3.00e8


def hkl_reciprocal_to_energy(hkl: torch.Tensor, reciprocal: torch.Tensor, **kwargs) -> torch.Tensor:
    """Alias to ``laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy``."""
    _, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal, _return_uq=False, **kwargs)
    return energy


def hkl_reciprocal_to_uq(hkl: torch.Tensor, reciprocal: torch.Tensor, **kwargs) -> torch.Tensor:
    """Alias to ``laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy``."""
    u_q, _ = hkl_reciprocal_to_uq_energy(hkl, reciprocal, _return_energy=False, **kwargs)
    return u_q


def hkl_reciprocal_to_uq_energy(
    hkl: torch.Tensor, reciprocal: torch.Tensor,
    *, cartesian_product: bool = True, _return_uq: bool = True, _return_energy: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Thanks to the bragg relation, compute the energy of each diffracted ray.

    Parameters
    ----------
    hkl : torch.Tensor
        The h, k, l indices of shape (\*n, 3) we want to mesure.
    reciprocal : torch.Tensor
        Matrix \(\mathbf{B}\) of shape (\*r, 3, 3) in the lab base \(\mathcal{B^l}\).
    cartesian_product : boolean, default=True
        If True (default value), batch dimensions are iterated independently like neasted for loop.
        Overwise, the batch dimensions are broadcasted like a zip.

        * True: The final shape are (\*n, \*r, 3) and (\*n, \*r).
        * False: The final shape are (\*broadcast(n, r), 3) and broadcast(n, r).

    Returns
    -------
    u_q : torch.Tensor
        All the unitary diffracting plane normal vector of shape (..., 3).
        The vectors are expressed in the same base as the reciprocal space.
    energy : torch.Tensor
        The energy of each ray in J as a tensor of shape (...).
        \(\begin{cases}
            E = \frac{hc}{\lambda} \\
            \lambda = 2d\sin(\theta) \\
            \sin(\theta) = \left| \langle u_i, u_q \rangle \right| \\
        \end{cases}\)

    Notes
    -----
    * According the pyfai convention, \(u_i = \mathbf{L_1}\).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.bragg import hkl_reciprocal_to_uq_energy
    >>> reciprocal = torch.torch.tensor([[ 1.6667e+09,  0.0000e+00,  0.0000e+00],
    ...                                  [ 9.6225e+08,  3.0387e+09, -0.0000e+00],
    ...                                  [-6.8044e+08, -2.1488e+09,  8.1653e+08]])
    >>> hkl = torch.tensor([1, 2, -1])
    >>> u_q, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal)
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
    assert isinstance(cartesian_product, bool), cartesian_product.__class__.__name__

    if cartesian_product:
        *batch_r, _, _ = reciprocal.shape
        *batch_n, _ = hkl.shape
        reciprocal = reciprocal[*((None,)*len(batch_n)), ..., :, :]  # (*n, *r, 3, 3)
        hkl = hkl[..., *((None,)*len(batch_r)), :]  # (*n, *r, 3)

    hkl = hkl[..., :, None]  # (..., 3, 1)
    u_q = reciprocal @ hkl.to(reciprocal.device, reciprocal.dtype)  # (..., 3, 1)
    inv_d_square = u_q.mT @ u_q  # (..., 1, 1)
    u_q = u_q.squeeze(-1)  # (..., 3)
    inv_d_square = inv_d_square.squeeze(-1)  # (..., 1)

    energy = None
    if _return_energy:
        ui_dot_uq = u_q[..., 2]  # (...)
        inv_d_sin_theta = inv_d_square.squeeze(-1) / ui_dot_uq
        energy = torch.abs((0.5 * PLANK_H * CELERITY_C) * inv_d_sin_theta)

    if _return_uq:
        u_q = u_q * torch.rsqrt(inv_d_square)

    return u_q, energy


def uf_to_uq(u_f: torch.Tensor) -> torch.Tensor:
    r"""Calculate the vector normal to the diffracting planes.

    Bijection of ``laueimproc.geometry.bragg.uq_to_uf``.

    \(u_q \propto u_f - u_i\)

    Parameters
    ----------
    u_f : torch.Tensor
        The unitary diffracted rays of shape (..., 3) in the lab base \(\mathcal{B^l}\).

    Returns
    -------
    u_q : torch.Tensor
        The unitary normals of shape (..., 3) in the lab base \(\mathcal{B^l}\).

    Notes
    -----
    * According the pyfai convention, \(u_i = \mathbf{L_1}\).
    """
    assert isinstance(u_f, torch.Tensor), u_f.__class__.__name__
    assert u_f.shape[-1:] == (3,), u_f.shape

    u_q = u_f.clone()
    u_q[..., 2] -= 1.0  # uf - ui
    norm = torch.sum(u_q * u_q, dim=-1, keepdim=True)
    # eps = torch.finfo(u_q.dtype).eps  # to solve case no deviation
    # norm[norm <= eps] = torch.inf  # to solve case no deviation
    u_q = u_q * torch.rsqrt(norm)
    return u_q


def uq_to_uf(u_q: torch.Tensor) -> torch.Tensor:
    r"""Calculate the diffracted ray from q vector.

    Bijection of ``laueimproc.geometry.bragg.uf_to_uq``.

    \(\begin{cases}
        u_f - u_i = \eta u_q \\
        \eta = 2 \langle u_q, -u_i \rangle \\
    \end{cases}\)

    Parameters
    ----------
    u_q : torch.Tensor
        The unitary q vectors of shape (..., 3) in the lab base \(\mathcal{B^l}\).

    Returns
    -------
    u_f : torch.Tensor
        The unitary diffracted ray of shape (..., 3) in the lab base \(\mathcal{B^l}\).

    Notes
    -----
    * \(u_f\) is not sensitive to the \(u_q\) orientation.
    * According the pyfai convention, \(u_i = \mathbf{L_1}\).
    """
    assert isinstance(u_q, torch.Tensor), u_q.__class__.__name__
    assert u_q.shape[-1:] == (3,), u_q.shape

    uq_dot_ui = u_q[..., 2]  # -<uq, -ui>
    q_norm = -2.0 * uq_dot_ui
    u_f = q_norm.unsqueeze(-1) * u_q
    u_f[..., 2] += 1.0  # eta * uq + ui
    return u_f
