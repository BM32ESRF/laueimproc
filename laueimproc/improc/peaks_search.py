#!/usr/bin/env python3

"""Atomic function for finding spots."""

import math
import numbers
import typing

import cv2
import numpy as np
import torch

from laueimproc.classes.tensor import Tensor



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
    bins = torch.logspace(
        start=-5.0, end=math.log10(torch.max(img).item()+1e-5), steps=101, device=img.device
    )
    bins -= 1e-5  # borns are [0, max(img)]
    hist = torch.histogram(img, bins=bins, density=False).hist
    hist /= float(img.shape[0]*img.shape[1])
    hist = torch.flip(hist, (0,))  # from white to black
    hist = torch.cumsum(hist, 0, out=hist)  # repartition function
    hist = torch.flip(hist, (0,))  # from black to white
    real_density = (
        density**(math.log(0.01)/math.log(0.5))  # make density linear [0, 0.5, 1] -> [0, 0.01, 1]
        * math.pi * 0.25  # circle to square <=> bbox to spot
        * 0.25  # max 25% filled
    )
    threshold = bins[torch.argmin((hist >= real_density).to(torch.uint8)).item() + 1].item()
    return threshold


def unfold(img: Tensor, kernel_h: int, kernel_w: int) -> Tensor:
    """Return patched version of img with a stride of 1.

    Parameters
    ----------
    img : Tensor
        A 2d tensor of shape (h, w).
    kernel_h : int
        The size of the height of the kernel <= h.
    kernel_w : int
        The size of the width of the kernel <= w.

    Returns
    -------
    patched_verstion : Tensor
        An image of shape (h+1-kernel_h, w+1-kernel_w, kernel_h, kernel_w)
        with overlapping non contiguous references on undeground data.

    Notes
    -----
    Call the method `contiguous` if you want to write on it.
    """
    assert isinstance(img, Tensor), img.__class__.__name__
    assert img.ndim == 2, img.shape
    assert isinstance(kernel_h, numbers.Integral), kernel_h.__class__.__name__
    assert kernel_h <= img.shape[0], (kernel_h, img.shape)
    assert isinstance(kernel_w, numbers.Integral), kernel_w.__class__.__name__
    assert kernel_w <= img.shape[1], (kernel_w, img.shape)

    img = torch.squeeze(
        torch.nn.functional.pad(
            torch.unsqueeze(img, 0),  # not implemented for 2d, only for 3d in torch 2.2.1
            (kernel_w//2, kernel_w//2 + kernel_w%2 - 1, kernel_h//2, kernel_h//2 + kernel_h%2 - 1),
            mode="replicate",
        ).contiguous(),  # maybe contiguous is useless but it is a security
        0,
    )
    return torch.as_strided(
        img,
        (img.shape[0]+1-kernel_h, img.shape[1]+1-kernel_w, kernel_h, kernel_w),  # shape
        (*img.stride(), *img.stride()),  # stride a step of 1
    )


def estimate_background(
    brut_image: np.ndarray,
    kernel_font: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
) -> np.ndarray:
    """Estimate and Return the background of the image.

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
    bg_image = cv2.GaussianBlur(brut_image, kernel_font.shape, 0)
    # bg_image = cv2.medianBlur(brut_image, 5)  # 5 is the max size
    return cv2.morphologyEx(brut_image, dst=bg_image, op=cv2.MORPH_OPEN, kernel=kernel_font)


def peaks_search(
    brut_image: Tensor,
    density: float = 0.5,
    kernel_aglo: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    *, _check: bool = True,
    **kwargs,
) -> tuple[Tensor, Tensor]:
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
    rois_no_background : Tensor
        The tensor of each regions of interest without background, shape (n, h, w) and type float.
    bboxes : Tensor
        The positions of the corners point (0, 0) and the shape (i, j, h, w)
        of the roi of each spot in the brut_image, and the height and width of each roi.
        The shape is (n, 4) and the type is int.
    """
    assert isinstance(brut_image, Tensor), brut_image.__class__.__name__
    assert brut_image.ndim == 2, brut_image.shape
    assert isinstance(density, numbers.Real), density.__class__.__type__
    assert density > 0.0, density
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
        # 2 times faster than cv2.threshold(fg_image, thresh, 1, cv2.THRESH_BINARY)
        # 3 times faster with torch than numpy
        (fg_image > _density_to_threshold_torch(Tensor(fg_image), density)).view(np.uint8)
        # (fg_image > _density_to_threshold_numpy(fg_image, density)).view(np.uint8)
    )
    binary = cv2.dilate(binary, kernel_aglo, dst=bg_image, iterations=1)
    bboxes = [
        (i, j, h, w) for j, i, w, h in map(  # cv2 to numpy convention
            cv2.boundingRect,
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        ) if max(h, w) <= 200 # remove too big spots
    ]

    # vectorisation
    # bboxes = (
    #     Tensor(torch.tensor(bboxes, dtype=int, device=brut_image.device))
    #     if bboxes else
    #     Tensor(torch.empty((0, 4), dtype=int))
    # )
    # rois = Tensor(
    #     torch.zeros( # zeros 2 times faster than empty + fill 0
    #         (
    #             bboxes.shape[0],
    #             torch.max(bboxes[:, 2]).item() if bboxes.shape[0] else 1,
    #             torch.max(bboxes[:, 3]).item() if bboxes.shape[0] else 1,
    #         ),
    #         dtype=brut_image.dtype,
    #         device=brut_image.device,
    #     )
    # )
    # for index, (i, j, height, width) in enumerate(bboxes.tolist()):  # write the right values
    #     rois[index, :height, :width] = torch.from_numpy(fg_image[i:i+height, j:j+width])
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
        rois = Tensor(torch.empty((0, 1, 1), dtype=brut_image.dtype, device=brut_image.device))
        bboxes = Tensor(torch.empty((0, 4), dtype=int, device=brut_image.device))
    return rois, bboxes
