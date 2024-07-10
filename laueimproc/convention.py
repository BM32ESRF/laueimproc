#!/usr/bin/env python3

"""Provide tools for switching convention."""

import functools
import numbers

import numpy as np
import torch


def det_to_poni(det: torch.Tensor) -> torch.Tensor:
    """Convert a .det config into a .poni config.

    Bijection of ``laueimproc.convention.poni_to_det``.

    Parameters
    ----------
    det : torch.Tensor
        The 5 + 1 ordered .det calibration parameters as a tensor of shape (..., 6):

        * `dd`: The distance sample to the point of normal incidence in mm.
        * `xcen`: The x coordinate of point of normal incidence in pixel.
        * `ycen`: The y coordinate of point of normal incidence in pixel.
        * `xbet`: One of the angle of rotation in degrees.
        * `xgam`: The other angle of rotation in degrees.
        * `pixelsize`: The size of one pixel in mm.

    Returns
    -------
    poni : torch.Tensor
        The 6 ordered .poni calibration parameters as a tensor of shape (..., 6):

        * `dist`: The distance sample to the point of normal incidence in m.
        * `poni1`: The coordinate of the point of normal incidence along d1 in m.
        * `poni2`: The coordinate of the point of normal incidence along d2 in m.
        * `rot1`: The first rotation in radian.
        * `rot2`: The second rotation in radian.
        * `rot1`: The third rotation in radian.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import det_to_poni, lauetools_to_pyfai, or_to_lauetools
    >>> from laueimproc.geometry.projection import ray_to_detector
    >>> uf_or = torch.randn(1000, 3)
    >>> uf_or *= torch.rsqrt(uf_or.sum(dim=1, keepdim=True))
    >>> det = torch.tensor([77.0, 800.0, 1200.0, 20.0, 20.0, 0.08])
    >>>
    >>> # using lauetools
    >>> uf_pyfai = lauetools_to_pyfai(or_to_lauetools(uf_or))
    >>> poni = det_to_poni(det)
    >>> xy_improc, _ = ray_to_detector(uf_pyfai, poni)
    >>> x_improc, y_improc = xy_improc[:, 0], xy_improc[:, 1]
    >>> cond = (x_improc > -1) & (x_improc < 1) & (y_improc > -1) & (y_improc < 1)
    >>> x_improc, y_improc = x_improc[cond], y_improc[cond]
    >>>
    >>> # using laueimproc
    >>> try:
    ...     from LaueTools.LaueGeometry import calc_xycam
    ... except ImportError:
    ...     pass
    ... else:
    ...     x_tools, y_tools, _ = calc_xycam(
    ...         uf_or.numpy(force=True), det[:-1].numpy(force=True), pixelsize=float(det[5])
    ...     )
    ...     x_tools = torch.asarray(x_tools, dtype=x_improc.dtype)[cond]
    ...     y_tools = torch.asarray(y_tools, dtype=y_improc.dtype)[cond]
    ...     x_tools, y_tools = det[5] * 1e-3 * x_tools, det[5] * 1e-3 * y_tools
    ...     assert torch.allclose(x_improc, x_tools)
    ...     assert torch.allclose(y_improc, y_tools)
    ...
    >>>
    """
    assert isinstance(det, torch.Tensor), det.__class__.__name__
    assert det.shape[-1:] == (6,), det.shape
    dist = 1e-3 * det[..., 0]
    poni1 = 1e-3 * det[..., 5] * det[..., 1]
    poni2 = 1e-3 * det[..., 5] * det[..., 2]
    rot1 = torch.zeros(det.shape[:-1], dtype=det.dtype, device=det.device)
    rot2 = -0.5*torch.pi + torch.deg2rad(det[..., 3])
    rot3 = 0.5*torch.pi - torch.deg2rad(det[..., 4])
    return torch.cat([
        dist.unsqueeze(-1),
        poni1.unsqueeze(-1),
        poni2.unsqueeze(-1),
        rot1.unsqueeze(-1),
        rot2.unsqueeze(-1),
        rot3.unsqueeze(-1),
    ], dim=-1)


def ij_to_xy(
    array: torch.Tensor | np.ndarray,
    *,
    i: tuple[int, slice, type(Ellipsis)] | int | slice | type(Ellipsis),
    j: tuple[int, slice, type(Ellipsis)] | int | slice | type(Ellipsis),
) -> torch.Tensor | np.ndarray:
    """Switch the axis i and j, and append 1/2 to all values.

    * `ij`: Extension by continuity (N -> R) of the numpy convention (height, width).
    The first axis iterates on lines from top to bottom, the second on columns from left to right.
    The origin (i=0, j=0) correspond to the top left image corner of the top left pixel.
    It means that the center of the top left pixel has the coordinate (i=1/2, j=1/2).

    * `xy`: A transposition and a translation of the origin of the `ij` convention.
    The first axis iterates on columns from left to right, the second on lines from top to bottom.
    In an image, the point (x=1, y=1) correspond to the middle of the top left pixel.


    .. image:: ../../build/media/IMGConvIJXY.avif

    Parameters
    ----------
    array : torch.Tensor or np.ndarray
        The data in ij convention.
    i, j : tuple, int, slice or Ellipsis
        The indexing of the i subdata and j subdata.

    Returns
    -------
    array : torch.Tensor or np.ndarray
        A reference to the ij_array, with the axis converted in xy convention.

    Notes
    -----
    Input and output data are shared in place.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import ij_to_xy
    >>> array = torch.zeros((10, 2))
    >>> array[:, 0] = torch.linspace(0, 1, 10)  # i axis
    >>> array[:, 1] = torch.linspace(2, 1, 10)  # j axis
    >>> array
    tensor([[0.0000, 2.0000],
            [0.1111, 1.8889],
            [0.2222, 1.7778],
            [0.3333, 1.6667],
            [0.4444, 1.5556],
            [0.5556, 1.4444],
            [0.6667, 1.3333],
            [0.7778, 1.2222],
            [0.8889, 1.1111],
            [1.0000, 1.0000]])
    >>> ij_to_xy(array, i=(..., 0), j=(..., 1))
    tensor([[2.5000, 0.5000],
            [2.3889, 0.6111],
            [2.2778, 0.7222],
            [2.1667, 0.8333],
            [2.0556, 0.9444],
            [1.9444, 1.0556],
            [1.8333, 1.1667],
            [1.7222, 1.2778],
            [1.6111, 1.3889],
            [1.5000, 1.5000]])
    >>> _ is array  # inplace
    True
    >>>
    """
    assert isinstance(array, torch.Tensor | np.ndarray), array.__class__.__name__
    ydata = array[i] + 0.5  # copy
    xdata = array[j]  # reference
    xdata += 0.5  # reference
    array[i] = xdata  # copy, change inplace
    array[j] = ydata  # copy, change inplace
    return array


def ij_to_xy_decorator(
    i: tuple[int, slice, type(Ellipsis)] | int | slice | type(Ellipsis),
    j: tuple[int, slice, type(Ellipsis)] | int | slice | type(Ellipsis),
):
    """Append the argument conv to a function to allow user switching convention."""
    def decorator(func: callable):
        @functools.wraps(func)
        def decorated(*args, conv: str = "ij", **kwargs):
            assert isinstance(conv, str), conv.__class__.__name__
            assert conv in {"ij", "xy"}, conv
            array = func(*args, **kwargs)  # assumed to be in ij convention
            if conv != "ij":
                array = globals()[f"ij_to_{conv}"](array, i=i, j=j)
            return array
        return decorated
    return decorator


def lauetools_to_or(vect_lauetools: torch.Tensor, dim: numbers.Integral = -1) -> torch.Tensor:
    """Active convertion of the vectors from lauetools base to odile base.

    Bijection of ``laueimproc.convention.or_to_lauetools``.

    .. image:: ../../build/media/IMGLauetoolsOr.avif

    Parameters
    ----------
    vect_lauetools : torch.Tensor
        The vector of shape (..., 3, ...) in the lauetools orthonormal base.
    dim : int, default=-1
        The axis index of the non batch dimension, such that vect.shape[dim] = 3.

    Returns
    -------
    vect_or : torch.Tensor
        The input vect, with the axis converted in the odile base.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import lauetools_to_or, or_to_lauetools
    >>> vect_odile = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> lauetools_to_or(vect_odile, dim=0)
    tensor([[ 0, -1,  0, -1],
            [ 1,  0,  0,  1],
            [ 0,  0,  1,  1]])
    >>> or_to_lauetools(_, dim=0)
    tensor([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]])
    >>>
    """
    assert isinstance(vect_lauetools, torch.Tensor), vect_lauetools.__class__.__name__
    assert isinstance(dim, numbers.Integral), dim.__class__.__name__
    assert vect_lauetools.ndim >= 1 and vect_lauetools.shape[dim] == 3, vect_lauetools.shape
    comp1, comp2, comp3 = torch.movedim(vect_lauetools, dim, 0)
    vect_or = torch.cat([-comp2.unsqueeze(0), comp1.unsqueeze(0), comp3.unsqueeze(0)])
    vect_or = torch.movedim(vect_or, 0, dim)
    return vect_or


def lauetools_to_pyfai(vect_lauetools: torch.Tensor, dim: numbers.Integral = -1) -> torch.Tensor:
    """Active convertion of the vectors from lauetools base to pyfai base.

    Bijection of ``laueimproc.convention.pyfai_to_lauetools``.

    .. image:: ../../build/media/IMGPyfaiLauetools.avif

    Parameters
    ----------
    vect_lauetools : torch.Tensor
        The vector of shape (..., 3, ...) in the lauetools orthonormal base.
    dim : int, default=-1
        The axis index of the non batch dimension, such that vect.shape[dim] = 3.

    Returns
    -------
    vect_pyfai : torch.Tensor
        The input vect, with the axis converted in the pyfai base.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import lauetools_to_pyfai, pyfai_to_lauetools
    >>> vect_pyfai = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> lauetools_to_pyfai(vect_pyfai, dim=0)
    tensor([[ 0,  0,  1,  1],
            [ 0, -1,  0, -1],
            [ 1,  0,  0,  1]])
    >>> pyfai_to_lauetools(_, dim=0)
    tensor([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]])
    >>>
    """
    assert isinstance(vect_lauetools, torch.Tensor), vect_lauetools.__class__.__name__
    assert isinstance(dim, numbers.Integral), dim.__class__.__name__
    assert vect_lauetools.ndim >= 1 and vect_lauetools.shape[dim] == 3, vect_lauetools.shape
    comp1, comp2, comp3 = torch.movedim(vect_lauetools, dim, 0)
    vect_pyfai = torch.cat([comp3.unsqueeze(0), -comp2.unsqueeze(0), comp1.unsqueeze(0)])
    vect_pyfai = torch.movedim(vect_pyfai, 0, dim)
    return vect_pyfai


def or_to_lauetools(vect_or: torch.Tensor, dim: numbers.Integral = -1) -> torch.Tensor:
    """Active convertion of the vectors from odile base to lauetools base.

    Bijection of ``laueimproc.convention.lauetools_to_or``.

    .. image:: ../../build/media/IMGLauetoolsOr.avif

    Parameters
    ----------
    vect_or : torch.Tensor
        The vector of shape (..., 3, ...) in the odile orthonormal base.
    dim : int, default=-1
        The axis index of the non batch dimension, such that vect.shape[dim] = 3.

    Returns
    -------
    vect_lauetools : torch.Tensor
        The input vect, with the axis converted in the lauetools base.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import lauetools_to_or, or_to_lauetools
    >>> vect_or = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> or_to_lauetools(vect_or, dim=0)
    tensor([[ 0,  1,  0,  1],
            [-1,  0,  0, -1],
            [ 0,  0,  1,  1]])
    >>> lauetools_to_or(_, dim=0)
    tensor([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]])
    >>>
    """
    assert isinstance(vect_or, torch.Tensor), vect_or.__class__.__name__
    assert isinstance(dim, numbers.Integral), dim.__class__.__name__
    assert vect_or.ndim >= 1 and vect_or.shape[dim] == 3, vect_or.shape
    comp1, comp2, comp3 = torch.movedim(vect_or, dim, 0)
    vect_lauetools = torch.cat([comp2.unsqueeze(0), -comp1.unsqueeze(0), comp3.unsqueeze(0)])
    vect_lauetools = torch.movedim(vect_lauetools, 0, dim)
    return vect_lauetools


def poni_to_det(poni: torch.Tensor, pixelsize: torch.Tensor) -> torch.Tensor:
    """Convert a .det config into a .poni config.

    Bijection of ``laueimproc.convention.det_to_poni``.

    Parameters
    ----------
    poni : torch.Tensor
        The 6 ordered .poni calibration parameters as a tensor of shape (..., 6):

        * `dist`: The distance sample to the point of normal incidence in m.
        * `poni1`: The coordinate of the point of normal incidence along d1 in m.
        * `poni2`: The coordinate of the point of normal incidence along d2 in m.
        * `rot1`: The first rotation in radian.
        * `rot2`: The second rotation in radian.
        * `rot1`: The third rotation in radian.
    pixelsize : torch.Tensor
        The size of one pixel in mm of shape (...).

    Returns
    -------
    det : torch.Tensor
        The 5 + 1 ordered .det calibration parameters as a tensor of shape (..., 6):

        * `dd`: The distance sample to the point of normal incidence in mm.
        * `xcen`: The x coordinate of point of normal incidence in pixel.
        * `ycen`: The y coordinate of point of normal incidence in pixel.
        * `xbet`: One of the angle of rotation in degrees.
        * `xgam`: The other angle of rotation in degrees.
        * `pixelsize`: The size of one pixel in mm.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import det_to_poni, poni_to_det
    >>> det = torch.tensor([77.0, 800.0, 1200.0, 20.0, 20.0, 0.08])
    >>> torch.allclose(det, poni_to_det(det_to_poni(det), det[5]))
    True
    >>>
    """
    assert isinstance(poni, torch.Tensor), poni.__class__.__name__
    assert poni.shape[-1:] == (6,), poni.shape
    assert isinstance(pixelsize, torch.Tensor), pixelsize.__class__.__name__
    assert poni.shape[:-1] == pixelsize.shape

    dd_ = 1e3 * poni[..., 0]
    xcen = poni[..., 1] / (pixelsize * 1e-3)
    ycen = poni[..., 2] / (pixelsize * 1e-3)
    xbet = torch.rad2deg(0.5*torch.pi + poni[..., 4])
    xgam = torch.rad2deg(0.5*torch.pi - poni[..., 5])

    return torch.cat([
        dd_.unsqueeze(-1),
        xcen.unsqueeze(-1),
        ycen.unsqueeze(-1),
        xbet.unsqueeze(-1),
        xgam.unsqueeze(-1),
        pixelsize.unsqueeze(-1),
    ], dim=-1)


def pyfai_to_lauetools(vect_pyfai: torch.Tensor, dim: numbers.Integral = -1) -> torch.Tensor:
    """Active convertion of the vectors from pyfai base to lauetools base.

    Bijection of ``laueimproc.convention.lauetools_to_pyfai``.

    .. image:: ../../build/media/IMGPyfaiLauetools.avif

    Parameters
    ----------
    vect_pyfai : torch.Tensor
        The vector of shape (..., 3, ...) in the pyfai orthonormal base.
    dim : int, default=-1
        The axis index of the non batch dimension, such that vect.shape[dim] = 3.

    Returns
    -------
    vect_lauetools : torch.Tensor
        The input vect, with the axis converted in the lauetools base.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import lauetools_to_pyfai, pyfai_to_lauetools
    >>> vect_pyfai = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pyfai_to_lauetools(vect_pyfai, dim=0)
    tensor([[ 0,  0,  1,  1],
            [ 0, -1,  0, -1],
            [ 1,  0,  0,  1]])
    >>> lauetools_to_pyfai(_, dim=0)
    tensor([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]])
    >>>
    """
    assert isinstance(vect_pyfai, torch.Tensor), vect_pyfai.__class__.__name__
    assert isinstance(dim, numbers.Integral), dim.__class__.__name__
    assert vect_pyfai.ndim >= 1 and vect_pyfai.shape[dim] == 3, vect_pyfai.shape
    comp1, comp2, comp3 = torch.movedim(vect_pyfai, dim, 0)
    vect_lauetools = torch.cat([comp3.unsqueeze(0), -comp2.unsqueeze(0), comp1.unsqueeze(0)])
    vect_lauetools = torch.movedim(vect_lauetools, 0, dim)
    return vect_lauetools
