#!/usr/bin/env python3

"""Convertion of diffracted ray to angles.

Do not use these functions for intensive calculation,
they should only be used for final conversion, not for a calculation step.
"""

import torch


def thetachi_to_uf(theta: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    r"""Reconstruct the diffracted ray from the deviation angles.

    Bijection of ``laueimproc.geometry.thetachi.uf_to_thetachi``.

    .. image:: ../../../build/media/IMGThetaChi.avif

    Parameters
    ----------
    theta : torch.Tensor
        The half deviation angle in radian, with \(\theta \in [0, \frac{\pi}{2}]\).
    chi : torch.Tensor
        The rotation angle in radian from the vertical plan \((\mathbf{L_1}, \mathbf{L_3})\)
        to the plan \((u_i, \mathcal{B^l_z})\), with \(\chi \in [-\pi, \pi]\).

    Returns
    -------
    u_f : torch.Tensor
        The unitary diffracted ray of shape (*broadcast(theta.shape, chi.shape), 3).
        It is expressed in the lab base \(\mathcal{B^l}\).

    Notes
    -----
    * According the pyfai convention, \(u_i = \mathbf{L_1}\).
    * This function is slow, use ``laueimproc.geometry.bragg.uq_to_uf`` if you can.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.thetachi import thetachi_to_uf
    >>> theta = torch.deg2rad(torch.tensor([15., 30., 45.]))
    >>> chi = torch.deg2rad(torch.tensor([ -0.,  90., -45.]))
    >>> thetachi_to_uf(theta, chi).round(decimals=4)
    tensor([[ 0.5000,  0.0000,  0.8660],
            [-0.0000, -0.8660,  0.5000],
            [ 0.7071,  0.7071, -0.0000]])
    >>>
    """
    assert isinstance(theta, torch.Tensor), theta.__class__.__name__
    assert isinstance(chi, torch.Tensor), chi.__class__.__name__

    batch = torch.broadcast_shapes(theta.shape, chi.shape)
    twicetheta = 2.0 * theta
    proj3 = torch.broadcast_to(torch.cos(twicetheta), batch)
    proj12 = torch.sin(twicetheta)
    proj1 = proj12 * torch.cos(chi)
    proj2 = proj12 * torch.sin(-chi)

    return torch.cat([proj1.unsqueeze(-1), proj2.unsqueeze(-1), proj3.unsqueeze(-1)], dim=-1)


def uf_to_thetachi(u_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Find the angular deviation of the dffracted ray.

    Bijection of ``laueimproc.geometry.thetachi.thetachi_to_uf``.

    .. image:: ../../../build/media/IMGThetaChi.avif

    Parameters
    ----------
    u_f : torch.Tensor
        The unitary diffracted ray of shape (..., 3) in the lab base \(\mathcal{B^l}\).

    Returns
    -------
    theta : torch.Tensor
        The half deviation angle in radian, of shape (...,). \(\theta \in [0, \frac{\pi}{2}]\)
    chi : torch.Tensor
        The counterclockwise (trigonometric) rotation of the diffracted ray if you look as u_i.
        It is the angle from the vertical plan \((\mathbf{L_1}, \mathbf{L_3})\)
        to the plan \((u_i, \mathcal{B^l_z})\).
        The shape is (...,) as well. \(\chi \in [-\pi, \pi]\)

    Notes
    -----
    * According the pyfai convention, \(u_i = \mathbf{L_1}\).
    * This function is slow, use ``laueimproc.geometry.bragg.uf_to_uq`` if you can.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.thetachi import uf_to_thetachi
    >>> u_f = torch.tensor([[1/2, 0, 3**(1/2)/2], [0, -3**(1/2)/2, 1/2], [2**(1/2), 2**(1/2), 0]])
    >>> theta, chi = uf_to_thetachi(u_f)
    >>> torch.rad2deg(theta).round()
    tensor([15., 30., 45.])
    >>> torch.rad2deg(chi).round()
    tensor([ -0.,  90., -45.])
    >>>
    """
    assert isinstance(u_f, torch.Tensor), u_f.__class__.__name__
    assert u_f.shape[-1:] == (3,), u_f.shape

    theta = 0.5 * torch.acos(u_f[..., 2])  # cos(2 * theta) = <uf, ui>
    chi = torch.atan2(-u_f[..., 1], u_f[..., 0])

    return theta, chi
