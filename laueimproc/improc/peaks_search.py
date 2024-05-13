#!/usr/bin/env python3

"""Atomic function for finding spots."""

import math
import numbers
import typing

import cv2
import numpy as np
import torch

from laueimproc.opti.rois import imgbboxes2raw
from laueimproc.improc.find_bboxes import find_bboxes
from laueimproc.improc.morpho import morpho_open


DEFAULT_KERNEL_AGLO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


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
    brut_image: torch.Tensor, radius_font: numbers.Real = 9, **_kwargs
) -> torch.Tensor:
    """Estimate and Return the background of the image.

    Parameters
    ----------
    brut_image : torch.Tensor
        The 2d array brut image of the Laue diagram in float between 0 and 1.
    radius_font : float, default = 9
        The structurant element radius used for the morphological opening.
        If it is not provided a circle of diameter 19 pixels is taken.
        More the radius or the kernel is big, better the estimation is,
        but slower the proccess is.

    Returns
    -------
    background : np.ndarray
        An estimation of the background.
    """
    # verifications
    assert isinstance(brut_image, torch.Tensor), brut_image.__class__.__name__
    assert brut_image.ndim == 2, brut_image.shape

    # extraction of the background
    bg_image = torch.from_numpy(
        cv2.medianBlur(brut_image.numpy(force=True), 5)  # 5 is the max size
    ).to(brut_image.device)
    # from scipy import signal
    # bg_image = signal.medfilt2d(brut_image, 5)  # no limitation but very slow!
    # bg_image = cv2.morphologyEx(brut_image, dst=bg_image, op=cv2.MORPH_OPEN, kernel=kernel_font)
    bg_image = morpho_open(bg_image, radius=radius_font, **_kwargs)
    # ksize = (3*kernel_font.shape[0], 3*kernel_font.shape[1])
    # bg_image = cv2.GaussianBlur(brut_image, dst=bg_image, ksize=ksize, sigmaX=0)
    bg_image = torch.minimum(bg_image, brut_image, out=bg_image)  # avoid < 0 when sub
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
    brut_image : torch.Tensor
        The 2d grayscale float32 image to annalyse.
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
    assert brut_image.dtype == torch.float32, brut_image.dtype
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
    # src = brut_image.numpy(force=True)  # not nescessary copy
    bg_image = estimate_background(brut_image, **kwargs)
    fg_image = brut_image - bg_image  # copy to keep brut_image unchanged
    binary = (fg_image > _density_to_threshold_torch(fg_image, density)).view(torch.uint8)
    binary = cv2.dilate(binary.numpy(force=True), kernel_aglo, iterations=1)
    bboxes = find_bboxes(binary).to(brut_image.device)
    datarois = imgbboxes2raw(fg_image, bboxes)
    return datarois, bboxes
