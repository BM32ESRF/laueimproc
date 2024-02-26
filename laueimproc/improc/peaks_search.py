#!/usr/bin/env python3

"""Atomic function for finding spots."""

import numbers
import typing

import cv2
import numpy as np
import torch

from laueimproc.classes.tensor import Tensor



DEFAULT_KERNEL_FONT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
DEFAULT_KERNEL_AGLO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))



def estimate_background(
    brut_image: np.ndarray,
    kernel_font: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
) -> np.ndarray:
    """Estimate and Return the background of the image.

    The method is based on morphological opening.

    Parameters
    ----------
    brut_image : Tensor
        The 2d array brut image of the Laue diagram in float between 0 and 1.
    kernel_font : np.ndarray[np.uint8, np.uint8], optional
        The structurant element used for the morphological opening.
        If it is not provided a circle of diameter 21 pixels is taken.
        More the size or the kernel is important, better the estimation is,
        but slower the proccess is.

    Returns
    -------
    background : np.ndarray
        An estimation of the background.
    """
    # verifications
    assert isinstance(brut_image, np.ndarray), brut_image.__class__.__name__
    assert brut_image.ndim == 2, brut_image.shape
    if kernel_font is None:
        kernel_font = DEFAULT_KERNEL_FONT
    elif _check:
        assert isinstance(kernel_font, np.ndarray), kernel_font.__class__.__name__
        assert kernel_font.dtype == np.uint8, kernel_font.dtype
        assert kernel_font.ndim == 2, kernel_font.shape
        assert kernel_font.shape[0] % 2 and kernel_font.shape[1] % 2, \
            f"the kernel has to be odd, current shape is {kernel_font.shape}"

    # extraction of the background
    return cv2.morphologyEx(brut_image, cv2.MORPH_OPEN, kernel_font)


def peaks_search(
    brut_image: Tensor,
    threshold: typing.Optional[float] = None,
    kernel_aglo: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Search all the spots roi in this diagram, return the roi tensor.

    Parameters
    ----------
    threshold : float, optional
        Only keep the peaks having a max intensity divide by the standard deviation
        of the image without background > threshold.
        If it is not provide, it takes a threshold that catch a lot of peaks.
    kernel_aglo : np.ndarray[np.uint8, np.uint8], optional
        The structurant element for the aglomeration of close grain
        by morhpological dilatation applied on the thresholed image.
        If it is not provide, it takes a circle of diameter 5.
        Bigger is the kernel, higher are the number of aglomerated spots.
    **kwargs : dict
        Transmitted to ``estimate_background``.

    Returns
    -------
    rois_no_background : Tensor
        The tensor of each regions of interest without background, shape (n, h, w) and type float.
    bboxes : Tensor
        The positions of the corners point (0, 0) and the shape (i, j, h, w)
        of the roi of each spot in the brut_image, and the height and width of each roi.
        The shape is (n, 4) and the type is int.
    """
    assert isinstance(brut_image, Tensor), brut_image.__class__.__name__
    assert brut_image.ndim == 2, brut_image.shape
    if threshold is not None:
        assert isinstance(threshold, numbers.Real), threshold.__class__.__type__
        assert threshold > 0.0, threshold
        threshold = float(threshold)
    else:
        threshold = 3.0
    if kernel_aglo is None:
        kernel_aglo = DEFAULT_KERNEL_AGLO
    else:
        assert isinstance(kernel_aglo, np.ndarray), threshold.__class__.__type__
        assert kernel_aglo.dtype == np.uint8, kernel_aglo.dtype
        assert kernel_aglo.ndim == 2, kernel_aglo.shape
        assert kernel_aglo.shape[0] % 2 and kernel_aglo.shape[1] % 2, \
            f"the kernel has to be odd, current shape is {kernel_aglo.shape}"

    # peaks search
    src = brut_image.numpy(force=True)  # not nescessary copy
    bg_image = estimate_background(src, **kwargs, _check=_check)
    if brut_image.data_ptr() == src.__array_interface__["data"][0]:
        fg_image = src - bg_image  # copy to keep self unchanged
    else:
        fg_image = src
        fg_image -= bg_image  # inplace
    bboxes = [
        (i, j, h, w) for j, i, w, h in map(  # cv2 to numpy convention
            cv2.boundingRect,
            cv2.findContours(
                cv2.dilate(
                    (fg_image > threshold * fg_image.std()).view(np.uint8),
                    # (fg_image > threshold).view(np.uint8),  # absolute threshold
                    kernel_aglo,
                    dst=bg_image,
                    iterations=1,
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0],
        ) if max(h, w) <= 200 # remove too big spots
    ]

    # vectorisation
    if bboxes:
        rois = np.zeros( # zeros 2 times faster than empty + fill 0
            (len(bboxes), max(h for _, _, h, _ in bboxes), max(w for _, _, _, w in bboxes)),
            dtype=fg_image.dtype
        )
        for index, (i, j, height, width) in enumerate(bboxes):  # write the right values
            rois[index, :height, :width] = fg_image[i:i+height, j:j+width]
        rois = Tensor(rois, to_float=True).to(device=brut_image.device)
        bboxes = Tensor(torch.tensor(bboxes, dtype=int))
    else:
        rois = Tensor(torch.empty((0, 1, 1), dtype=torch.float32, device=brut_image.device))
        bboxes = Tensor(torch.empty((0, 4), dtype=int))
    return rois, bboxes
