#!/usr/bin/env python3

"""Crop and pad an image to schange the size without any interpolation."""

import numbers

import numpy as np
import torch


def _crop(image: torch.Tensor, cropping: tuple[int, int, int, int], copy: bool) -> torch.Tensor:
    """Remove borders to reduce the image size.

    Parameters
    ----------
    image : torch.Tensor
        The image to be padded, of shape (height, width).
    cropping : tuple[int, int, int, int]
        The pixel size of borders top, bottom, left, right.
    copy : boolean
        If True, ensure that the returned tensor doesn't share the data of the input tensor.

    Returns
    -------
    cropped_image : torch.Tensor
        The cropped image.

    Notes
    -----
    * No verifications are performed for performance reason.
    * The output tensor can be a reference to the provided tensor if copy is False.
    """
    # optimization, avoid useless memory copy
    if cropping == (0, 0, 0, 0):
        return image.clone() if copy else image

    # preparation
    height, width = image.shape
    top, bottom, left, right = cropping

    # remove borders
    cropped_image = image[top:height-bottom, left:width-right]
    if copy:
        cropped_image = cropped_image.clone()
    return cropped_image


def _pad(image: torch.Tensor, padding: tuple[int, int, int, int], copy: bool) -> torch.Tensor:
    """Add black borders to enlarge an image.

    Parameters
    ----------
    image : torch.Tensor
        The image to be padded, of shape (height, width).
    padding : tuple[int, int, int, int]
        The pixel size of borders top, bottom, left, right.
    copy : boolean
        If True, ensure that the returned tensor doesn't share the data of the input tensor.

    Returns
    -------
    padded_image : torch.Tensor
        The padded image.

    Notes
    -----
    * No verifications are performed for performance reason.
    * The output tensor can be a reference to the provided tensor if copy is False.
    """
    # optimization, avoid useless memory copy
    if padding == (0, 0, 0, 0):
        return image.clone() if copy else image

    # preparation
    height, width = image.shape
    top, bottom, left, right = padding

    # manage alpha channel
    padded_image = torch.empty(   # 1200 times faster than torch.zeros
        (height+top+padding[1], width+padding[2]+padding[3]),
        dtype=image.dtype, device=image.device
    )

    # make black borders
    if top:
        padded_image[:top, :] = 0.0
    if bottom:
        padded_image[height+top:, :] = 0.0
    if left:
        padded_image[top:height+top, :left] = 0.0
    if right:
        padded_image[top:height+top, width+left:] = 0.0

    # copy image content
    padded_image[top:height+top, left:width+left] = image
    return padded_image


def _patch(image: torch.Tensor, shape: tuple[int, int], copy: bool) -> torch.Tensor:
    """Help `patch`.

    Notes
    -----
    * No verifications are performed for performance reason.
    * The output tensor can be a reference to the provided tensor if copy is False.
    """
    # cropping
    if image.shape[0] > shape[0]:
        top, bottom = divmod(image.shape[0]-shape[0], 2)
        bottom += top
    else:
        top, bottom = 0, 0
    if image.shape[1] > shape[1]:
        left, right = divmod(image.shape[1]-shape[1], 2)
        right += left
    else:
        left, right = 0, 0
    image = _crop(image, (top, bottom, left, right), copy=False)

    # padding
    if image.shape[0] < shape[0]:
        top, bottom = divmod(shape[0]-image.shape[0], 2)
        bottom += top
    else:
        top, bottom = 0, 0
    if image.shape[1] < shape[1]:
        left, right = divmod(shape[1]-image.shape[1], 2)
        right += left
    else:
        left, right = 0, 0
    image = _pad(image, (top, bottom, left, right), copy=copy)

    return image


def patch(
    image: torch.Tensor | np.ndarray,
    shape: tuple[numbers.Integral, numbers.Integral] | list[numbers.Integral],
    copy: bool = True,
) -> torch.Tensor | np.ndarray:
    """Pad the image with transparent borders.

    Parameters
    ----------
    image : torch.Tensor or numpy.ndarray
        The image to be cropped and padded.
    shape : int and int
        The pixel dimensions of the returned image.
        The convention adopted is the numpy convention (height, width).
    copy : boolean, default=True
        If True, ensure that the returned tensor doesn't share the data of the input tensor.

    Returns
    -------
    patched_image
        The cropped and padded image homogeneous with the input.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.nn.dataaug.patch import patch
    >>> ref = torch.full((4, 8), 128, dtype=torch.uint8)
    >>> patch(ref, (6, 6))
    tensor([[  0,   0,   0,   0,   0,   0],
            [128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128],
            [  0,   0,   0,   0,   0,   0]], dtype=torch.uint8)
    >>> patch(ref, (3, 9))
    tensor([[128, 128, 128, 128, 128, 128, 128, 128,   0],
            [128, 128, 128, 128, 128, 128, 128, 128,   0],
            [128, 128, 128, 128, 128, 128, 128, 128,   0]], dtype=torch.uint8)
    >>> patch(ref, (2, 2))
    tensor([[128, 128],
            [128, 128]], dtype=torch.uint8)
    >>> patch(ref, (10, 10))
    tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0, 128, 128, 128, 128, 128, 128, 128, 128,   0],
            [  0, 128, 128, 128, 128, 128, 128, 128, 128,   0],
            [  0, 128, 128, 128, 128, 128, 128, 128, 128,   0],
            [  0, 128, 128, 128, 128, 128, 128, 128, 128,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=torch.uint8)
    >>>
    """
    assert isinstance(image, torch.Tensor | np.ndarray), image.__class__.__name__

    # case cast homogeneous
    if isinstance(image, np.ndarray):
        return patch(torch.from_numpy(image), shape, copy=copy).numpy(force=True)

    # cast shape
    if image.ndim > 2 and image.shape[0] == 1:
        return torch.unsqueeze(patch(torch.squeeze(image, 0), shape, copy=copy), 0)

    # verif
    assert image.ndim == 2, image.shape
    assert image.shape[0] >= 1, image.shape
    assert image.shape[1] >= 1, image.shape
    assert isinstance(shape, tuple | list), shape.__class__.__name__
    assert len(shape) == 2, len(shape)
    assert all(isinstance(s, numbers.Integral) for s in shape), shape
    assert shape >= (1, 1), shape
    shape = (int(shape[0]), int(shape[1]))
    assert isinstance(copy, bool), copy.__class__.__name__

    # pad
    return _patch(image, shape, copy=copy)
