#!/usr/bin/env python3

"""Resize and image keeping the proportions."""

import numbers

import cv2
import numpy as np
import torch

from laueimproc.nn.dataaug.patch import _patch


def _resize(image: torch.Tensor, shape: tuple[int, int], copy: bool) -> torch.Tensor:
    """Help ``resize``.

    Notes
    -----
    * No verifications are performed for performance reason.
    * The output tensor can be a reference to the provided tensor if copy is False.
    """
    if image.shape == shape:  # optional optimization
        return image.clone() if copy else image
    height, width = shape
    enlarge = height >= image.shape[0] or width >= image.shape[1]
    image_np = image.numpy(force=True)
    image_np = np.ascontiguousarray(image_np)  # cv2 needs it
    image_np = cv2.resize(  # 10 times faster than torchvision.transforms.v2.functional.resize
        image_np,
        dsize=(width, height),
        interpolation=(cv2.INTER_CUBIC if enlarge else cv2.INTER_AREA),  # for antialiasing
    )
    if enlarge and np.issubdtype(image_np.dtype, np.floating):
        image_np = np.clip(image_np, 0.0, 1.0, out=image_np)
    image = torch.from_numpy(image_np).to(image.device)
    return image


def rescale(
    image: torch.Tensor | np.ndarray,
    shape: tuple[numbers.Integral, numbers.Integral] | list[numbers.Integral],
    copy: bool = True,
) -> torch.Tensor | np.ndarray:
    """Reshape the image, keep the spact ratio and pad with black pixels.

    Parameters
    ----------
    image : cutcutcodec.core.classes.image_video.FrameVideo or torch.Tensor or numpy.ndarray
        The image to be resized, of shape (height, width).
    shape : int and int
        The pixel dimensions of the returned image.
        The convention adopted is the numpy convention (height, width).
    copy : boolean, default=True
        If True, ensure that the returned tensor doesn't share the data of the input tensor.

    Returns
    -------
    resized_image
        The resized (and padded) image homogeneous with the input.
        The underground data are not shared with the input. A safe copy is done.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.nn.dataaug.scale import rescale
    >>> ref = torch.full((4, 8), 128, dtype=torch.uint8)
    >>>
    >>> # upscale
    >>> rescale(ref, (8, 12))
    tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
           dtype=torch.uint8)
    >>>
    >>> # downscale
    >>> rescale(ref, (4, 4))
    tensor([[  0,   0,   0,   0],
            [128, 128, 128, 128],
            [128, 128, 128, 128],
            [  0,   0,   0,   0]], dtype=torch.uint8)
    >>>
    >>> # mix
    >>> rescale(ref, (6, 6))
    tensor([[  0,   0,   0,   0,   0,   0],
            [128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128],
            [  0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0]], dtype=torch.uint8)
    >>>
    """
    assert isinstance(image, (torch.Tensor, np.ndarray)), image.__class__.__name__

    # case cast homogeneous
    if isinstance(image, np.ndarray):
        return rescale(torch.from_numpy(image), shape, copy=copy).numpy(force=True)

    # cast shape
    if image.ndim > 2 and image.shape[0] == 1:
        return torch.unsqueeze(rescale(torch.squeeze(image, 0), shape, copy=copy), 0)

    # verif
    assert image.ndim == 2, image.shape
    assert image.shape[0] >= 1, image.shape
    assert image.shape[1] >= 1, image.shape
    assert isinstance(shape, (tuple, list)), shape.__class__.__name__
    assert len(shape) == 2, len(shape)
    assert all(isinstance(s, numbers.Integral) for s in shape), shape
    assert shape >= (1, 1), shape
    shape = (int(shape[0]), int(shape[1]))
    assert isinstance(copy, bool), copy.__class__.__name__

    # find the shape for keeping proportion
    dw_sh, dh_sw = shape[1]*image.shape[0], shape[0]*image.shape[1]
    if dw_sh < dh_sw:  # need vertical padding
        height, width = (round(dw_sh/image.shape[1]), shape[1])  # keep width unchanged
    elif dw_sh > dh_sw:  # need horizontal padding
        height, width = (shape[0], round(dh_sw/image.shape[0]))  # keep height unchanged
    else:  # if the proportion is the same
        return _resize(image, shape, copy=copy)

    # resize and pad
    image = _resize(image, (height, width), copy=copy)
    image = _patch(image, shape, copy=False)
    return image
