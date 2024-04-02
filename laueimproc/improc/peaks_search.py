#!/usr/bin/env python3

"""Atomic function for finding spots."""

import math
import numbers
import typing

import cv2
import numpy as np
import torch

from laueimproc.io.read import to_floattensor
from laueimproc.opti.rois import imgbboxes2raw


DEFAULT_KERNEL_FONT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))  # 17 - 21
DEFAULT_KERNEL_AGLO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _density_to_threshold_numpy(img: np.ndarray, density: float) -> float:
    """Analyse the image histogram in order to find the correspondant threshold."""
    hist, bin_edges = np.histogram(
        img,
        bins=np.logspace(-5.0, np.log10(np.max(img)+1e-5), 101) - 1e-5,  # borns are [0, max(img)]
        density=False,
    )
    hist = hist.astype(float) / float(img.size)  # density hist
    cum_hist = np.flip(np.cumsum(np.flip(hist), out=hist))  # repartition function
    real_density = (
        density**(np.log(0.01)/np.log(0.5))  # make density linear [0, 0.5, 1] -> [0, 0.01, 1]
        * np.pi * 0.25  # circle to square <=> bbox to spot
        * 0.25  # max 25% filled
    )
    threshold = bin_edges[np.argmin(cum_hist >= real_density) + 1]
    return threshold


def _density_to_threshold_torch(img: torch.Tensor, density: float) -> float:
    """Analyse the image histogram in order to find the correspondant threshold."""
    device = img.device
    bins = torch.logspace(
        start=-5.0, end=math.log10(torch.max(img).item()+1e-5), steps=101, device=device
    )
    bins -= 1e-5  # borns are [0, max(img)]
    hist = torch.histogram(img.to("cpu"), bins=bins.to("cpu"), density=False).hist.to(device)
    hist /= float(img.shape[0]*img.shape[1])
    hist = torch.flip(hist, (0,))  # from white to black
    hist = torch.cumsum(hist, 0, out=hist)  # repartition function
    hist = torch.flip(hist, (0,))  # from black to white
    real_density = (
        density**(math.log(0.01)/math.log(0.5))  # make density linear [0, 0.5, 1] -> [0, 0.01, 1]
        * math.pi * 0.25  # circle to square <=> bbox to spot
        * 0.25  # max 25% filled
    )
    threshold = bins[torch.argmin((hist >= real_density).view(torch.uint8)).item() + 1].item()
    return threshold


def estimate_background(
    brut_image: np.ndarray,
    kernel_font: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
) -> np.ndarray:
    """Estimate and Return the background of the image.

    Parameters
    ----------
    brut_image : torch.Tensor
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
    bg_image = cv2.medianBlur(brut_image, 5)  # 5 is the max size
    # from scipy import signal
    # bg_image = signal.medfilt2d(brut_image, 5)  # no limitation but very slow!
    bg_image = cv2.morphologyEx(brut_image, dst=bg_image, op=cv2.MORPH_OPEN, kernel=kernel_font)
    # ksize = (3*kernel_font.shape[0], 3*kernel_font.shape[1])
    # bg_image = cv2.GaussianBlur(brut_image, dst=bg_image, ksize=ksize, sigmaX=0)
    bg_image = np.clip(bg_image, a_min=0.0, a_max=brut_image, out=bg_image)  # avoid 0 when substract
    return bg_image


def peaks_search(
    brut_image: torch.Tensor,
    density: float = 0.5,
    kernel_aglo: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search all the spots roi in this diagram, return the roi tensor.

    Parameters
    ----------
    density : float, default=0.5
        Correspond to the density of spots found.
        This value is normalized so that it can evolve 'linearly' between ]0, 1].
        The smaller the value, the fewer spots will be captured.
    kernel_aglo : np.ndarray[np.uint8, np.uint8], optional
        The structurant element for the aglomeration of close grain
        by morhpological dilatation applied on the thresholed image.
        If it is not provide, it takes a circle of diameter 5.
        Bigger is the kernel, higher are the number of aglomerated spots.
    **kwargs : dict
        Transmitted to ``estimate_background``.

    Returns
    -------
    rois_no_background : bytearray
        The flatten float32 tensor data of each regions of interest without background.
        After unfolding and padding, the shape is (n, h, w).
    bboxes : torch.Tensor
        The positions of the corners point (0, 0) and the shape (i, j, h, w)
        of the roi of each spot in the brut_image, and the height and width of each roi.
        The shape is (n, 4) and the type is int.
    """
    assert isinstance(brut_image, torch.Tensor), brut_image.__class__.__name__
    assert brut_image.ndim == 2, brut_image.shape
    assert isinstance(density, numbers.Real), density.__class__.__type__
    assert 0.0 < density <= 1.0, density
    density = float(density)
    if kernel_aglo is None:
        kernel_aglo = DEFAULT_KERNEL_AGLO
    else:
        assert isinstance(kernel_aglo, np.ndarray), kernel_aglo.__class__.__type__
        assert kernel_aglo.dtype == np.uint8, kernel_aglo.dtype
        assert kernel_aglo.ndim == 2, kernel_aglo.shape
        assert kernel_aglo.shape[0] % 2 and kernel_aglo.shape[1] % 2, \
            f"the kernel has to be odd, current shape is {kernel_aglo.shape}"

    # peaks search
    src = brut_image.numpy(force=True)  # not nescessary copy
    bg_image = estimate_background(src, **kwargs, _check=_check)
    if brut_image.data_ptr() == src.__array_interface__["data"][0]:
        fg_image = src - bg_image  # copy to keep brut_image unchanged
    else:
        fg_image = src
        fg_image -= bg_image  # inplace
    binary = (
        (fg_image > _density_to_threshold_torch(torch.from_numpy(fg_image), density)).view(np.uint8)
    )
    binary = cv2.dilate(binary, kernel_aglo, dst=bg_image, iterations=1)
    bboxes = [
        (i, j, h, w) for j, i, w, h in map(  # cv2 to numpy convention
            cv2.boundingRect,
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        ) if max(h, w) <= 200 # remove too big spots
    ]

    if bboxes:
        bboxes = torch.tensor(bboxes, dtype=torch.int32)
    else:
        bboxes = torch.empty((0, 4), dtype=torch.int32, device=brut_image.device)
    datarois = imgbboxes2raw(torch.from_numpy(fg_image), bboxes)
    return datarois, bboxes
