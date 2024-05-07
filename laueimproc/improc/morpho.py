#!/usr/bin/env python3

"""Morphological image operation with rond kernels."""

import functools
import logging
import math
import numbers

import cv2
import torch

try:
    from laueimproc.improc import c_morpho
except ImportError:
    logging.warning(
        "failed to import laueimproc.improc.c_find_bboxes, a slow python version is used instead"
    )
    c_morpho = None


@functools.lru_cache(2)
def get_circle_kernel(radius: numbers.Real) -> torch.Tensor:
    """Compute a circle structurant element.

    Parameters
    ----------
    radius : float
        The radius of the circle, such as diameter is 2*radius.

    Return
    ------
    circle : torch.Tensor
        The float32 2d image of the circle.
        The dimension size dim is odd.

    Notes
    -----
    Not equivalent to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius, 2*radius)).

    Examples
    --------
    >>> from laueimproc.improc.morpho import get_circle_kernel
    >>> get_circle_kernel(0)
    tensor([[1]], dtype=torch.uint8)
    >>> get_circle_kernel(1)
    tensor([[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=torch.uint8)
    >>> get_circle_kernel(2)
    tensor([[0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]], dtype=torch.uint8)
    >>> get_circle_kernel(3)
    tensor([[0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0]], dtype=torch.uint8)
    >>> get_circle_kernel(6)
    tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    >>> get_circle_kernel(2.5)
    tensor([[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=torch.uint8)
    >>>
    """
    assert isinstance(radius, numbers.Real), radius.__class__.__name__
    assert radius >= 0, radius

    int_radius = math.ceil(radius - 0.5)
    pos = torch.arange(-int_radius, int_radius+1, dtype=torch.float32)
    pos = torch.meshgrid(pos, pos, indexing="ij")
    kernel = (pos[0]*pos[0] + pos[1]*pos[1]) <= radius*radius
    kernel = kernel.view(torch.uint8)
    return kernel


def morphology(
    image: torch.Tensor, radius: numbers.Real, operation: str, *, _no_c: bool = False
) -> torch.Tensor:
    """Apply the morphological operation on the image.

    Parameters
    ----------
    image : torch.Tensor
        The float32 c contiguous 2d image.
    radius : float
        The radius of the structurant rond kernel.
    operation : str
        The operation type, can be "close", "dilate", "erode", "open".

    Returns
    -------
    image : torch.Tensor
        The transformed image as a reference to input image.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.morpho import morphology
    >>> image = torch.rand((2048, 2048))
    >>> out = morphology(image.clone(), radius=3, operation="open")
    >>> torch.equal(out, morphology(image.clone(), radius=3, operation="open", _no_c=True))
    True
    >>>
    """
    assert isinstance(image, torch.Tensor), image.__class__.__name__
    image_np = image.numpy(force=True)

    if not _no_c and c_morpho is not None and image.shape[1] >= 4:
        c_func = {
            "close": c_morpho.close,
            "dilate": c_morpho.dilate,
            "erode": c_morpho.erode,
            "open": c_morpho.open,
        }[operation]
        return torch.from_numpy(c_func(image_np, radius)).to(device=image.device)

    assert image.ndim == 2, image.shape
    assert image.dtype == torch.float32, image.dtype
    assert image.is_contiguous(), "image has to be c contiguous"
    assert isinstance(radius, numbers.Real), radius.__class__.__name__
    assert radius >= 0, radius
    assert isinstance(operation, str), operation.__class__.__name__
    assert operation in {"close", "dilate", "erode", "open"}, operation

    kernel = get_circle_kernel(radius).numpy(force=True)
    operation_cv2 = {
        "close": cv2.MORPH_OPEN,
        "dilate": cv2.MORPH_DILATE,
        "erode": cv2.MORPH_ERODE,
        "open": cv2.MORPH_OPEN,
    }[operation]
    image_np = cv2.morphologyEx(image_np, dst=image_np, op=operation_cv2, kernel=kernel)
    image = torch.from_numpy(image_np).to(device=image.device)
    return image


def morpho_close(image: torch.Tensor, radius: numbers.Real, *, _no_c: bool = False) -> torch.Tensor:
    """Apply a morphological closing on the image.

    Parameters
    ----------
    image : torch.Tensor
        The float32 c contiguous 2d image.
    radius : float
        The radius of the structurant rond kernel.

    Returns
    -------
    image : torch.Tensor
        The closed image as a reference to the input image.
    """
    return morphology(image, radius, "close", _no_c=_no_c)


def morpho_dilate(
    image: torch.Tensor, radius: numbers.Real, *, _no_c: bool = False
) -> torch.Tensor:
    """Apply a morphological dilatation on the image.

    Parameters
    ----------
    image : torch.Tensor
        The float32 c contiguous 2d image.
    radius : float
        The radius of the structurant rond kernel.

    Returns
    -------
    image : torch.Tensor
        The dilated image as a reference to the input image.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.morpho import morpho_dilate
    >>> image = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ...                       [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    >>> morpho_dilate(image, radius=2)
    tensor([[0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
            [0., 0., 1., 0., 0., 0., 0., 1., 1., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 1.],
            [1., 1., 1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 1., 1., 1., 0., 0., 1., 1., 1.]])
    >>>
    """
    return morphology(image, radius, "dilate", _no_c=_no_c)


def morpho_erode(image: torch.Tensor, radius: numbers.Real, *, _no_c: bool = False) -> torch.Tensor:
    """Apply a morphological erosion on the image.

    Parameters
    ----------
    image : torch.Tensor
        The float32 c contiguous 2d image.
    radius : float
        The radius of the structurant rond kernel.

    Returns
    -------
    image : torch.Tensor
        The eroded image as a reference to the input image.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.morpho import morpho_erode
    >>> image = torch.tensor([[0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    ...                       [0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    ...                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                       [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    ...                       [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    ...                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                       [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    ...                       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    ...                       [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]], dtype=torch.float32)
    >>> morpho_erode(image, radius=2)
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 1., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 1.]])
    >>>
    """
    return morphology(image, radius, "erode", _no_c=_no_c)


def morpho_open(image: torch.Tensor, radius: numbers.Real, *, _no_c: bool = False) -> torch.Tensor:
    """Apply a morphological opening on the image.

    Parameters
    ----------
    image : torch.Tensor
        The float32 c contiguous 2d image.
    radius : float
        The radius of the structurant rond kernel.

    Returns
    -------
    image : torch.Tensor
        The opend image as a reference to the input image.
    """
    return morphology(image, radius, "open", _no_c=_no_c)
