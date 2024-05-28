#!/usr/bin/env python3

"""Manage a compact representation of the rois."""

import logging

import torch

try:
    from laueimproc.opti import c_rois
except ImportError:
    logging.warning(
        "failed to import laueimproc.opti.c_rois, a slow python version is used instead"
    )
    c_rois = None


def filter_by_indices(
    indices: torch.Tensor, data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False
) -> tuple[bytearray, torch.Tensor]:
    """Select the rois of the given indices.

    Parameters
    ----------
    indices : torch.Tensor
        The 1d int64 list of the rois index to keep. Negative indexing is allow.
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4).
        It doesn't have to be c contiguous.

    Returns
    -------
    filtered_data : bytearray
        The new flatten rois sorted according the provided indices.
    filtered_bboxes : bytearray
        The new reorganized bboxes.

    Notes
    -----
    When using the c backend, the negative indices of `indices` are set inplace as positive value.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from laueimproc.opti.rois import filter_by_indices
    >>> indices = torch.tensor([1, 1, 0, -1, -2, *range(100, 1000)])
    >>> bboxes = torch.zeros((1000, 4), dtype=torch.int16)
    >>> bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40
    >>> data = bytearray(
    ...     np.linspace(0, 1, (bboxes[:, 2]*bboxes[:, 3]).sum(), dtype=np.float32).tobytes()
    ... )
    >>> new_data, new_bboxes = filter_by_indices(indices, data, bboxes)
    >>> new_bboxes.shape
    torch.Size([905, 4])
    >>> assert new_data == filter_by_indices(indices, data, bboxes, _no_c=True)[0]
    >>> assert torch.all(new_bboxes == filter_by_indices(indices, data, bboxes, _no_c=True)[1])
    >>>
    """
    if not _no_c and c_rois is not None:
        indices_np = indices.numpy(force=True)
        bboxes_np = bboxes.numpy(force=True)
        new_data, new_bboxes_np = c_rois.filter_by_indices(indices_np, data, bboxes_np)
        return new_data, torch.from_numpy(new_bboxes_np).to(bboxes.dtype)

    assert isinstance(indices, torch.Tensor), indices.__class__.__name__
    assert indices.ndim == 1, indices.shape
    assert indices.dtype == torch.int64, indices.__class__.__name__
    assert isinstance(data, bytearray), data.__class__.__name__
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape
    assert bboxes.dtype == torch.int16, bboxes.dtype
    assert torch.all(bboxes[:, 2:] >= 1), "some bboxes have area of zero"
    assert len(data) == torch.float32.itemsize*(bboxes[:, 2]*bboxes[:, 3]).sum(), \
        "data length dosen't match rois area"

    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)
    new_rois, new_bboxes = rois[indices], bboxes[indices]
    new_data = roisshapes2raw(new_rois, new_bboxes[:, 2:], _no_c=_no_c)
    return new_data, new_bboxes


def imgbboxes2raw(img: torch.Tensor, bboxes: torch.Tensor, *, _no_c: bool = False) -> bytearray:
    """Extract the rois from the image.

    Parameters
    ----------
    img : torch.Tensor
        The float32 grayscale image of a laue diagram of shape (h, w).
        It doesn't have to be c contiguous but it is faster if it is the case.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4).
        It doesn't have to be c contiguous.

    Returns
    -------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.opti.rois import imgbboxes2raw
    >>> img = torch.rand((2000, 2000))
    >>> bboxes = torch.zeros((1000, 4), dtype=torch.int16)
    >>> bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40
    >>> data = imgbboxes2raw(img, bboxes)
    >>> len(data)
    2200000
    >>> assert data == imgbboxes2raw(img, bboxes, _no_c=True)
    >>>
    """
    if not _no_c and c_rois is not None:
        img_np = img.numpy(force=True)
        bboxes_np = bboxes.numpy(force=True)
        return c_rois.imgbboxes2raw(img_np, bboxes_np)

    assert isinstance(img, torch.Tensor), img.__class__.__name__
    assert img.ndim == 2, img.shape
    assert img.dtype == torch.float32, img.dtype
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape
    assert bboxes.dtype == torch.int16, bboxes.dtype
    assert torch.all(bboxes[:, 2:] >= 1), "some bboxes have area of zero"
    assert torch.all(bboxes[:, :2] >= 0), "some bboxes come out of the picture"
    assert torch.all(bboxes[:, 0] + bboxes[:, 2] <= img.shape[0]), \
        "some bboxes come out of the picture"
    assert torch.all(bboxes[:, 1] + bboxes[:, 3] <= img.shape[1]), \
        "some bboxes come out of the picture"

    if len(bboxes) == 0:
        return bytearray(b"")
    flat_rois = torch.cat(
        [img[a_h:a_h+s_h, a_w:a_w+s_w].ravel() for a_h, a_w, s_h, s_w in bboxes.tolist()]
    )
    data = bytearray(flat_rois.numpy(force=True).tobytes())
    return data


def rawshapes2rois(
    data: bytearray, shapes: torch.Tensor, *, _no_c: bool = False
) -> torch.Tensor:
    """Unfold and pad the flatten rois data into a tensor.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
    shapes : torch.Tensor
        The int16 tensor that contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.

    Returns
    -------
    rois : torch.Tensor
        The unfolded and padded rois of dtype torch.float32 and shape (n, h, w).

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from laueimproc.opti.rois import rawshapes2rois
    >>> shapes = torch.zeros((1000, 2), dtype=torch.int16)
    >>> shapes[::2, 0], shapes[1::2, 0], shapes[::2, 1], shapes[1::2, 1] = 10, 20, 30, 40
    >>> data = bytearray(
    ...     np.linspace(0, 1, (shapes[:, 0]*shapes[:, 1]).sum(), dtype=np.float32).tobytes()
    ... )
    >>> rois = rawshapes2rois(data, shapes)
    >>> rois.shape
    torch.Size([1000, 20, 40])
    >>> assert np.array_equal(rois, rawshapes2rois(data, shapes, _no_c=True))
    >>>
    """
    if not _no_c and c_rois is not None:
        shapes_np = shapes.numpy(force=True)
        rois_np = c_rois.rawshapes2rois(data, shapes_np)
        return torch.from_numpy(rois_np).to(shapes.device)

    assert isinstance(data, bytearray), data.__class__.__name__
    assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__
    assert shapes.ndim == 2, shapes.shape
    assert shapes.shape[1] == 2, shapes.shape
    assert shapes.dtype == torch.int16, shapes.dtype
    assert torch.all(shapes >= 1), "some bboxes have area of zero"
    shapes_32 = shapes.to(torch.int32)  # for overflow
    assert len(data) == torch.float32.itemsize*(shapes_32[:, 0]*shapes_32[:, 1]).sum(), \
        "data length dosen't match rois area"

    if not data:
        return torch.empty((0, 1, 1), dtype=torch.float32, device=shapes.device)
    flat_rois = torch.frombuffer(data, dtype=torch.float32).to(shapes.device)
    rois = torch.zeros(
        (len(shapes), shapes[:, 0].max(), shapes[:, 1].max()),
        dtype=torch.float32, device=shapes.device,
    )
    ptr = 0
    for i, (height, width) in enumerate(shapes.tolist()):
        next_ptr = ptr + height*width
        rois[i, :height, :width] = flat_rois[ptr:next_ptr].reshape(height, width)
        ptr = next_ptr
    return rois


def roisshapes2raw(
    rois: torch.Tensor, shapes: torch.Tensor, *, _no_c: bool = False
) -> bytearray:
    """Compress the rois into a flatten no padded respresentation.

    Parameters
    ----------
    rois : torch.Tensor
        The unfolded and padded rois of dtype torch.float32 and shape (n, h, w).
    shapes : torch.Tensor
        The int16 tensor that contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.

    Returns
    -------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.opti.rois import roisshapes2raw
    >>> shapes = torch.zeros((1000, 2), dtype=torch.int16)
    >>> shapes[::2, 0], shapes[1::2, 0], shapes[::2, 1], shapes[1::2, 1] = 10, 20, 30, 40
    >>> rois = torch.zeros((1000, 20, 40), dtype=torch.float32)
    >>> for i, (h, w) in enumerate(shapes.tolist()):
    ...     rois[i, :h, :w] = (i+1)/1000
    ...
    >>> data = roisshapes2raw(rois, shapes)
    >>> len(data)
    2200000
    >>> assert data == roisshapes2raw(rois, shapes, _no_c=True)
    >>>
    """
    if not _no_c and c_rois is not None:
        rois_np = rois.numpy(force=True)
        shapes_np = shapes.numpy(force=True)
        return c_rois.roisshapes2raw(rois_np, shapes_np)

    assert isinstance(rois, torch.Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert rois.shape[1:] >= (1, 1), rois.shape
    assert rois.dtype == torch.float32, rois.dtype
    assert isinstance(shapes, torch.Tensor), shapes.__class__.__name__
    assert shapes.ndim == 2, shapes.shape
    assert shapes.shape[1] == 2, shapes.shape
    assert shapes.dtype == torch.int16, shapes.dtype
    assert torch.all(shapes >= 1), "some bboxes have area of zero"
    assert len(rois) == len(shapes), (rois.shape, shapes.shape)

    if len(shapes) == 0:  # torch.cat dosen't support empty list
        return bytearray(b"")
    flat_rois = torch.cat(  # rois is float32
        [rois[i, :h, :w].ravel() for i, (h, w) in enumerate(shapes.tolist())]
    )
    data = bytearray(flat_rois.numpy(force=True).tobytes())
    return data
