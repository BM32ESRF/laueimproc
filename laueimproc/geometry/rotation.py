#!/usr/bin/env python3

"""Help for making rotation matrix."""

import numbers

import torch


def omega_to_rot(
    omega1: torch.Tensor | numbers.Real = 0,
    omega2: torch.Tensor | numbers.Real = 0,
    omega3: torch.Tensor | numbers.Real = 0,
    *, cartesian_product: bool = True,
) -> torch.Tensor:
    r"""Generate a rotation matrix from the given angles.

    The rotation are following the pyfai convention.

    Parameters
    ----------
    omega1 : torch.Tensor or float
        The first rotation angle of shape (\*a,).
        \(
            rot_1 =
            \begin{pmatrix}
                1 & 0              &               0 \\
                0 & \cos(\omega_1) & -\sin(\omega_1) \\
                c & \sin(\omega_1) &  \cos(\omega_1) \\
            \end{pmatrix}
        \)
    omega2 : torch.Tensor or float
        The second rotation angle of shape (\*b,).
        \(
            rot_2 =
            \begin{pmatrix}
                 \cos(\omega_2) & 0 & \sin(\omega_2) \\
                              0 & 1 &              0 \\
                -\sin(\omega_2) & 0 & \cos(\omega_2) \\
            \end{pmatrix}
        \)
    omega3 : torch.Tensor or float
        The third rotation angle of shape (\*c,). (inverse of pyfai convention)
        \(
            rot_3 =
            \begin{pmatrix}
                \cos(\omega_3) & -\sin(\omega_3) & 0 \\
                \sin(\omega_3) &  \cos(\omega_3) & 0 \\
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

    Notes
    -----
    To be bijective with the function ``laueimproc.geometry.rotation.rot_to_omega``,
    you have to respect: \(
        [\omega_1, \omega_2, \omega_3]
        \in [-\pi, \pi] \times [-\frac{\pi}{2}, \frac{\pi}{2}] \times [-\pi, \pi]
    \)

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import omega_to_rot
    >>> omega_to_rot(omega1=torch.pi/6, omega2=torch.pi/6, omega3=torch.pi/6)
    tensor([[ 0.7500, -0.2165,  0.6250],
            [ 0.4330,  0.8750, -0.2165],
            [-0.5000,  0.4330,  0.7500]])
    >>> omega_to_rot(torch.randn(4), torch.randn(5, 6), torch.randn(7, 8, 9)).shape
    torch.Size([4, 5, 6, 7, 8, 9, 3, 3])
    >>>
    """
    assert isinstance(omega1, torch.Tensor | numbers.Real), omega1.__class__.__name__
    assert isinstance(omega2, torch.Tensor | numbers.Real), omega2.__class__.__name__
    assert isinstance(omega3, torch.Tensor | numbers.Real), omega3.__class__.__name__
    assert isinstance(cartesian_product, bool), cartesian_product.__class__.__name__

    # find dtype and device
    dtype = (
        {a.dtype for a in (omega1, omega2, omega3) if isinstance(a, torch.Tensor)}
        or {torch.float32}
    )
    dtype = max(dtype, key=lambda t: t.itemsize)
    device = (
        {a.device for a in (omega1, omega2, omega3) if isinstance(a, torch.Tensor)}
        or {torch.device("cpu")}
    ).pop()

    # cast into tensor
    if isinstance(omega1, numbers.Real):
        omega1 = torch.tensor(omega1, device=device, dtype=dtype)
    if isinstance(omega2, numbers.Real):
        omega2 = torch.tensor(omega2, device=device, dtype=dtype)
    if isinstance(omega3, numbers.Real):
        omega3 = torch.tensor(omega3, device=device, dtype=dtype)

    # expand dims
    if cartesian_product:
        omega1, omega2, omega3 = (
            omega1[..., *((None,)*omega2.ndim), *((None,)*omega3.ndim)],
            omega2[*((None,)*omega1.ndim), ..., *((None,)*omega3.ndim)],
            omega3[*((None,)*omega1.ndim), *((None,)*omega2.ndim), ...],
        )
    batch = torch.broadcast_shapes(omega1.shape, omega2.shape, omega3.shape)

    # precompute sin and cos
    sin1, cos1 = torch.sin(omega1), torch.cos(omega1)
    sin2, cos2 = torch.sin(omega2), torch.cos(omega2)
    sin3, cos3 = torch.sin(omega3), torch.cos(omega3)

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


def rand_rot(nbr: numbers.Integral, **kwargs) -> torch.Tensor:
    """Draw a random uniform rotation matrix.

    Parameters
    ----------
    nbr : int
        The number of matrices to draw.
    **kwargs : dict
        Transmitted to torch.rand, it can be "device" or "dtype" for example.

    Returns
    -------
    rot : torch.Tensor
        Rotation matrices uniformly distributed in the space of possible rotations.
        No one direction is favored over another. Shape (nbr, 3, 3).

    Notes
    -----
    Creating rotation matrices from a uniform draw of omegas induces a bias
    that will favor certain oriantations.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import rand_rot
    >>> rot = rand_rot(1000, dtype=torch.float64)
    >>> torch.allclose(torch.linalg.det(rot), torch.tensor(1.0, dtype=rot.dtype))
    True
    >>> torch.allclose(rot @ rot.mT, torch.eye(3, dtype=rot.dtype))
    True
    >>>
    """
    assert isinstance(nbr, numbers.Integral), nbr.__class__.__name__
    assert nbr >= 0, nbr

    # draw 2*nbr random vectors in the unitary sphere, kind of Monte-Carlo:
    # we have (1 + 9%) * (Vcube / Vsphere) = 2, it means if we draw two time more vectors
    # than the required number, it is fine in average.
    vect = torch.empty((0, 3), **kwargs)
    norm = torch.empty((0,), **kwargs)
    while len(vect) < 2*nbr:
        new_vect = torch.rand(2 * (2*nbr - len(vect)), 3, **kwargs)
        new_vect *= 2.0
        new_vect -= 1.0  # uniform in [-1, 1[
        new_norm = torch.sum(new_vect * new_vect, dim=1)
        cond = new_norm <= 1.0  # the proba of keeping it is Vshere / Vcube = (4/3*pi) / (2*2*2)
        vect, norm = torch.cat([vect, new_vect[cond]]), torch.cat([norm, new_norm[cond]])
    vect, norm = vect[:2*nbr], norm[:2*nbr]

    # create orthonormal base with Gram-Schmidt
    vect1 = vect[:nbr]
    vect1 *= torch.rsqrt(norm[:nbr, None])
    vect2 = vect[nbr:]
    vect2 -= vect1 * torch.sum(vect1 * vect2, dim=1, keepdim=True)  # <vect1, vect2>
    vect2 *= torch.rsqrt(torch.sum(vect2 * vect2, dim=1, keepdim=True))
    vect3 = torch.linalg.cross(vect1, vect2)

    # concatenate it as rotation matrices
    return torch.cat([vect1[:, :, None], vect2[:, :, None], vect3[:, :, None]], dim=2)


def rot_to_omega(rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Extract the rotation angles from a fulle rotation matrix.

    Bijection of ``laueimproc.geometry.rotation.omega_to_rot``.

    Parameters
    ----------
    rot : torch.Tensor
        The rotation matrix \(rot_3 . rot_2 . rot_1\) of shape (..., 3, 3).

    Returns
    -------
    omega1 : torch.Tensor or float
        The first rotation angle of shape (...,). \(\omega_1 \in [-\pi, \pi]\)
        \(
            rot_1 =
            \begin{pmatrix}
                1 & 0              &               0 \\
                0 & \cos(\omega_1) & -\sin(\omega_1) \\
                c & \sin(\omega_1) &  \cos(\omega_1) \\
            \end{pmatrix}
        \)
    omega2 : torch.Tensor or float
        The second rotation angle of shape (...,). \(\omega_2 \in [-\frac{\pi}{2}, \frac{\pi}{2}]\)
        \(
            rot_2 =
            \begin{pmatrix}
                 \cos(\omega_2) & 0 & \sin(\omega_2) \\
                              0 & 1 &              0 \\
                -\sin(\omega_2) & 0 & \cos(\omega_2) \\
            \end{pmatrix}
        \)
    omega3 : torch.Tensor or float
        The third rotation angle of shape (...,). (inverse of pyfai convention)
        \(\omega_3 \in [-\pi, \pi]\)
        \(
            rot_3 =
            \begin{pmatrix}
                \cos(\omega_3) & -\sin(\omega_3) & 0 \\
                \sin(\omega_3) &  \cos(\omega_3) & 0 \\
                             0 &               0 & 1 \\
            \end{pmatrix}
        \)

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import rot_to_omega
    >>> rot = torch.tensor([[ 0.7500, -0.2165,  0.6250],
    ...                     [ 0.4330,  0.8750, -0.2165],
    ...                     [-0.5000,  0.4330,  0.7500]])
    >>> theta = rot_to_omega(rot)
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
    omega1 = torch.atan2(rot[..., 2, 1], rot[..., 2, 2])
    omega2 = -torch.asin(rot[..., 2, 0])
    omega3 = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    return torch.cat([omega1.unsqueeze(-1), omega2.unsqueeze(-1), omega3.unsqueeze(-1)], dim=-1)


def rotate_crystal(
    crystal: torch.Tensor,
    rot: torch.Tensor,
    *, cartesian_product: bool = True,
) -> torch.Tensor:
    r"""Apply an active rotation to the crystal.

    Parameters
    ----------
    crystal : torch.Tensor
        The primitive \(\mathbf{A}\) or reciprocal \(\mathbf{B}\) in the base \(\mathcal{B}\).
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


def uniform_so3_meshgrid(
    resol: numbers.Real,
    omega_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    **kwargs,
) -> torch.Tensor:
    r"""Sample the omega space to be uniformly sampled in the rotation space.

    Parameters
    ----------
    resol : float
        The angular resolution in radian.
    omega_range : tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        The min and max limits for each elementary rotation angle
        \omega_1, \omega_2, \omega_3, in radian.

    Returns
    -------
    omega : torch.Tensor
        The 3 elementary rotation angles of shape (n, 3).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.geometry.rotation import uniform_so3_meshgrid
    >>> uniform_so3_meshgrid(
    ...     0.7 * torch.pi/180, ((0, torch.pi/2), (-torch.pi/2, torch.pi/2), (-torch.pi, torch.pi))
    ... )
    tensor([[ 0.0000, -1.5708, -3.1416],
            [ 0.0000, -1.5708, -3.1294],
            [ 0.0000, -1.5708, -3.1172],
            ...,
            [ 1.5638,  1.5237,  3.1137],
            [ 1.5638,  1.5237,  3.1259],
            [ 1.5638,  1.5237,  3.1381]])
    >>>
    """
    assert isinstance(resol, numbers.Real), resol.__class__.__name__
    assert resol > 2.926e-9, resol  # 2*pi / resol < 2**31 - 1 because cast int32
    assert isinstance(omega_range, tuple), omega_range.__name__.__name__
    assert len(omega_range) == 3, omega_range
    assert all(isinstance(mm, tuple) for mm in omega_range), omega_range
    assert all(len(mm) == 2 for mm in omega_range), omega_range
    (omega1_min, omega1_max), (omega2_min, omega2_max), (omega3_min, omega3_max) = omega_range
    assert omega1_min >= -torch.pi, omega1_min
    assert omega1_max <= torch.pi, omega1_max
    assert omega1_min < omega1_max, (omega1_min, omega1_max)
    assert omega2_min >= -0.5 * torch.pi, omega2_min
    assert omega2_max <= 0.5 * torch.pi, omega2_max
    assert omega2_min < omega2_max, (omega2_min, omega2_max)
    assert omega3_min >= -torch.pi, omega3_min
    assert omega3_max <= torch.pi, omega3_max
    assert omega3_min < omega3_max, (omega3_min, omega3_max)

    # If we draw a lot of uniform rotation matrix with rand_rot,
    # and we compute the histograms of the omega, we observe that
    # omega1 and omega3 are uniform, and the histogram of omega2 is a perfect cosinus.
    # So we have to apply an arcsin into uniform omega2' for having unifor rotation.

    all_omega1 = torch.arange(omega1_min, omega1_max, resol, **kwargs)
    all_omega2 = torch.asin(torch.arange(omega2_min, omega2_max, resol, **kwargs) / (0.5*torch.pi))
    all_omega3 = torch.arange(omega3_min, omega3_max, resol, **kwargs)

    all_omega = torch.meshgrid(all_omega1, all_omega2, all_omega3, indexing="ij")
    all_omega = torch.cat([o.reshape(-1, 1) for o in all_omega], dim=1)
    return all_omega
