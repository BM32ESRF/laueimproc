#!/usr/bin/env python3

"""Help for making rotation matrix."""

import numbers

import torch


def angle_to_rot(
    theta1: torch.Tensor | numbers.Real = 0,
    theta2: torch.Tensor | numbers.Real = 0,
    theta3: torch.Tensor | numbers.Real = 0,
    *, cartesian_product: bool = True,
) -> torch.Tensor:
    r"""Generate a rotation matrix from the given angles.

    The rotation are following the pyfai convention.

    Parameters
    ----------
    theta1 : torch.Tensor or float
        The first rotation angle of shape (\*a,).
        \(
            rot_1 =
            \begin{pmatrix}
                1 & 0              &               0 \\
                0 & \cos(\theta_1) & -\sin(\theta_1) \\
                c & \sin(\theta_1) &  \cos(\theta_1) \\
            \end{pmatrix}
        \)
    theta2 : torch.Tensor or float
        The second rotation angle of shape (\*b,).
        \(
            rot_2 =
            \begin{pmatrix}
                 \cos(\theta_2) & 0 & \sin(\theta_2) \\
                              0 & 1 &              0 \\
                -\sin(\theta_2) & 0 & \cos(\theta_2) \\
            \end{pmatrix}
        \)
    theta3 : torch.Tensor or float
        The third rotation angle of shape (\*c,). (inverse of pyfai convention)
        \(
            rot_3 =
            \begin{pmatrix}
                \cos(\theta_3) & -\sin(\theta_3) & 0 \\
                \sin(\theta_3) &  \cos(\theta_3) & 0 \\
                             0 &               0 & 1 \\
            \end{pmatrix}
        \)
    cartesian_product : boolean, default=True
        If True (default value), batch dimensions are iterated independently like neasted for loop.
        Overwise, the batch dimensions are broadcasted like a zip.

        * True: The final shape is (\*a, \*b, \*c, 3, 3).
        * False: The final shape is (\*broadcast(a, b, c), 3, 3).

    Returns
    -------
    rot : torch.Tensor
        The global rotation \(rot_3 . rot_2 . rot_1\).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import angle_to_rot
    >>> angle_to_rot(theta1=torch.pi/6, theta2=torch.pi/6, theta3=torch.pi/6)
    tensor([[ 0.7500, -0.2165,  0.6250],
            [ 0.4330,  0.8750, -0.2165],
            [-0.5000,  0.4330,  0.7500]])
    >>> angle_to_rot(torch.randn(4), torch.randn(5, 6), torch.randn(7, 8, 9)).shape
    torch.Size([4, 5, 6, 7, 8, 9, 3, 3])
    >>>
    """
    assert isinstance(theta1, torch.Tensor | numbers.Real), theta1.__class__.__name__
    assert isinstance(theta2, torch.Tensor | numbers.Real), theta2.__class__.__name__
    assert isinstance(theta3, torch.Tensor | numbers.Real), theta3.__class__.__name__
    assert isinstance(cartesian_product, bool), cartesian_product.__class__.__name__

    # find dtype and device
    dtype = (
        {a.dtype for a in (theta1, theta2, theta3) if isinstance(a, torch.Tensor)}
        or {torch.float32}
    )
    dtype = max(dtype, key=lambda t: t.itemsize)
    device = (
        {a.device for a in (theta1, theta2, theta3) if isinstance(a, torch.Tensor)}
        or {torch.device("cpu")}
    ).pop()

    # cast into tensor
    if isinstance(theta1, numbers.Real):
        theta1 = torch.tensor(theta1, device=device, dtype=dtype)
    if isinstance(theta2, numbers.Real):
        theta2 = torch.tensor(theta2, device=device, dtype=dtype)
    if isinstance(theta3, numbers.Real):
        theta3 = torch.tensor(theta3, device=device, dtype=dtype)

    # expand dims
    if cartesian_product:
        theta1, theta2, theta3 = (
            theta1[..., *((None,)*theta2.ndim), *((None,)*theta3.ndim)],
            theta2[*((None,)*theta1.ndim), ..., *((None,)*theta3.ndim)],
            theta3[*((None,)*theta1.ndim), *((None,)*theta2.ndim), ...],
        )
    batch = torch.broadcast_shapes(theta1.shape, theta2.shape, theta3.shape)

    # precompute sin and cos
    sin1, cos1 = torch.sin(theta1), torch.cos(theta1)
    sin2, cos2 = torch.sin(theta2), torch.cos(theta2)
    sin3, cos3 = torch.sin(theta3), torch.cos(theta3)

    # full matrix
    sin1sin2 = sin1 * sin2
    cos1sin2 = cos1 * sin2
    return torch.cat([
        (cos2*cos3).expand(batch).unsqueeze(-1),
        (sin1sin2*cos3 - cos1*sin3).expand(batch).unsqueeze(-1),
        (cos1sin2*cos3 + sin1*sin3).expand(batch).unsqueeze(-1),
        (cos2*sin3).expand(batch).unsqueeze(-1),
        (sin1sin2*sin3 + cos1*cos3).expand(batch).unsqueeze(-1),
        (cos1sin2*sin3 - sin1*cos3).expand(batch).unsqueeze(-1),
        (-sin2).expand(batch).unsqueeze(-1),
        (sin1*cos2).expand(batch).unsqueeze(-1),
        (cos1*cos2).expand(batch).unsqueeze(-1),
    ], dim=-1).reshape(*batch, 3, 3)


def rot_to_angle(rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Extract the rotation angles from a fulle rotation matrix.

    Bijection of ``laueimproc.geometry.rotation.angle_to_rot``.

    Parameters
    ----------
    rot : torch.Tensor
        The rotation matrix \(rot_3 . rot_2 . rot_1\) of shape (..., 3, 3).

    Returns
    -------
    theta1 : torch.Tensor or float
        The first rotation angle of shape (...,). \(\theta_1 \in [-\pi, \pi]\)
        \(
            rot_1 =
            \begin{pmatrix}
                1 & 0              &               0 \\
                0 & \cos(\theta_1) & -\sin(\theta_1) \\
                c & \sin(\theta_1) &  \cos(\theta_1) \\
            \end{pmatrix}
        \)
    theta2 : torch.Tensor or float
        The second rotation angle of shape (...,). \(\theta_2 \in [-\frac{\pi}{2}, \frac{\pi}{2}]\)
        \(
            rot_2 =
            \begin{pmatrix}
                 \cos(\theta_2) & 0 & \sin(\theta_2) \\
                              0 & 1 &              0 \\
                -\sin(\theta_2) & 0 & \cos(\theta_2) \\
            \end{pmatrix}
        \)
    theta3 : torch.Tensor or float
        The third rotation angle of shape (...,). (inverse of pyfai convention)
        \(\theta_3 \in [-\pi, \pi]\)
        \(
            rot_3 =
            \begin{pmatrix}
                \cos(\theta_3) & -\sin(\theta_3) & 0 \\
                \sin(\theta_3) &  \cos(\theta_3) & 0 \\
                             0 &               0 & 1 \\
            \end{pmatrix}
        \)

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import rot_to_angle
    >>> rot = torch.tensor([[ 0.7500, -0.2165,  0.6250],
    ...                     [ 0.4330,  0.8750, -0.2165],
    ...                     [-0.5000,  0.4330,  0.7500]])
    >>> theta = rot_to_angle(rot)
    >>> torch.rad2deg(theta[..., 0]).round()
    tensor(30.)
    >>> torch.rad2deg(theta[..., 1]).round()
    tensor(30.)
    >>> torch.rad2deg(theta[..., 2]).round()
    tensor(30.)
    >>>
    """
    assert isinstance(rot, torch.Tensor), rot.__class__.__name__
    assert rot.shape[-2:] == (3, 3), rot.shape
    theta1 = torch.atan2(rot[..., 2, 1], rot[..., 2, 2])
    theta2 = -torch.asin(rot[..., 2, 0])
    theta3 = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    return torch.cat([theta1.unsqueeze(-1), theta2.unsqueeze(-1), theta3.unsqueeze(-1)], dim=-1)


def rotate_crystal(
    crystal: torch.Tensor,
    rot: torch.Tensor,
    *, cartesian_product: bool = True,
) -> torch.Tensor:
    r"""Apply an active rotation to the crystal.

    Parameters
    ----------
    crystal : torch.Tensor
        The primitive \((\mathbf{A})\) or reciprocal \((\mathbf{B})\) in the base \(\mathcal{B}\).
        The shape of this parameter is (\*c, 3, 3).
    rot : torch.Tensor
        The active rotation matrix, of shape (\*r, 3, 3).
    cartesian_product : boolean, default=True
        If True (default value), batch dimensions are iterated independently like neasted for loop.
        Overwise, the batch dimensions are broadcasted like a zip.

        * True: The final shape is (\*c, \*r, 3, 3).
        * False: The final shape is (\*broadcast(c, r), 3, 3).

    Returns
    -------
    rotated_crystal: torch.Tensor
        The batched matricial product `rot @ crystal`.
    """
    assert isinstance(crystal, torch.Tensor), crystal.__class__.__name__
    assert crystal.shape[-2:] == (3, 3), crystal.shape
    assert isinstance(rot, torch.Tensor), rot.__class__.__name__
    assert rot.shape[-2:] == (3, 3), rot.shape
    assert isinstance(cartesian_product, bool), cartesian_product.__class__.__name__

    if cartesian_product:
        *batch_c, _, _ = crystal.shape
        *batch_r, _, _ = rot.shape
        crystal = crystal[..., *((None,)*len(batch_r)), :, :]
        rot = rot[*((None,)*len(batch_c)), ..., :, :]

    return rot @ crystal
