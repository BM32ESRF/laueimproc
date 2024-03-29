#!/usr/bin/env python3

"""Manage a compact representation of the rois."""

import logging

import numpy as np
import torch

try:
    from laueimproc.opti import c_rois
except ImportError:
    logging.warning(
        "failed to import laueimproc.opti.c_rois, a slow python version is used instead"
    )
    c_rois = None


def bin2rois(data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False) -> torch.Tensor:
    """Unfold and pad the flatten rois data into a tensor.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated float32 rois.
    bboxes : torch.Tensor
        Contains the information of the bboxes shapes.
        heights = bboxes[:, 2] and widths = bboxes[:, 3].

    Returns
    -------
    rois : torch.Tensor
        The unfolded and padded rois of dtype torch.float32 and shape (n, h, w).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.opti.rois import bin2rois
    >>> bboxes = torch.zeros((1000, 4), dtype=torch.int32)
    >>> bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40
    >>> data = bytearray(torch.linspace(0, 1, (bboxes[:, 2]*bboxes[:, 3]).sum()).numpy().tobytes())
    >>> rois = bin2rois(data, bboxes)
    >>> rois.shape
    torch.Size([1000, 20, 40])
    >>>
    """
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    flat_rois = np.frombuffer(data, np.float32)
    shapes = bboxes.numpy(force=True)[:, 2:]  # not c contiguous

    if not _no_c and c_rois is not None:
        return torch.from_numpy(c_rois.bin2rois(data, shapes))

    assert isinstance(data, bytearray), data.__class__.__name__
    assert shapes.dtype == np.int32, shapes.dtype
    assert np.all(shapes >= 1), "some shapes have area of zero"

    rois = np.zeros((len(shapes), shapes[:, 0].max(), shapes[:, 1].max()), dtype=np.float32)
    ptr = 0
    for i, (height, width) in enumerate(shapes.tolist()):
        next_ptr = ptr + height*width
        rois[i, :height, :width] = flat_rois[ptr:next_ptr].reshape(height, width)
        ptr = next_ptr
    return torch.from_numpy(rois)
