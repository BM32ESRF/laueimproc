#!/usr/bin/env python3

"""Give simple information on spots."""

import logging

import torch

try:
    from laueimproc.improc.spot import c_basic
except ImportError:
    logging.warning(
        "failed to import laueimproc.improc.spot.c_basic, a slow python version is used instead"
    )
    c_basic = None
from laueimproc.opti.rois import rawshapes2rois


def compute_barycenters(
    data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False
) -> torch.Tensor:
    """Find the weighted barycenter of each roi.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.

    Returns
    -------
    positions : torch.Tensor
        The position of the height and width barycenter for each roi of shape (n, 2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.spot.basic import compute_barycenters
    >>> patches = [
    ...     torch.tensor([[1.0]]),
    ...     torch.tensor([[1.0, 1.0], [1.0, 2.0]]),
    ...     torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
    ... ] * 2
    >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes())
    >>> bboxes = tensor([[ 0,  0,  1,  1],
    ...                  [ 0,  0,  2,  2],
    ...                  [ 0,  0,  5,  1],
    ...                  [10, 10,  1,  1],
    ...                  [10, 10,  2,  2],
    ...                  [10, 10,  5,  1]], dtype=torch.int16)
    >>> print(compute_barycenters(data, bboxes))
    >>> torch.allclose(
    ...     compute_barycenters(data, bboxes),
    ...     compute_barycenters(data, bboxes, _no_c=True),
    ... )
    True
    >>>
    """
    if not _no_c and c_basic is not None:
        bboxes_np = bboxes.numpy(force=True)
        barycenters_np = c_basic.compute_barycenters(data, bboxes_np)
        return torch.from_numpy(barycenters_np).to(bboxes.device)

    assert isinstance(data, bytearray), data.__class__.__name__
    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape
    assert bboxes.dtype == torch.int16, bboxes.dtype
    assert torch.all(bboxes[:, 2:] >= 1), "some bboxes have area of zero"
    assert len(data) == torch.float32.itemsize*(bboxes[:, 2]*bboxes[:, 3]).sum(), \
        "data length dosen't match rois area"

    rois = rawshapes2rois(data, bboxes[:, 2:])  # (n, h, w)
    _, height, width = rois.shape
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, height+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, width+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )  # (h, w)
    points = torch.cat([points_i.ravel().unsqueeze(0), points_j.ravel().unsqueeze(0)])  # (2, h*w)
    points = points.unsqueeze(0)  # (1, 2, h*w)
    weight = rois.reshape(-1, 1, height*width)  # (n, 1, h*w)
    pond_points = points * weight  # (n, 2, h*w)
    pond_points /= torch.sum(weight, axis=-1, keepdim=True)  # (n, 2, h*w)
    mean = torch.sum(pond_points, axis=-1)  # (n, 2)
    mean += bboxes[:, :2].to(rois.dtype)  # relative to absolute base
    return mean


def compute_rois_max(tensor_spots: torch.Tensor) -> torch.Tensor:
    """Return the intensity maxi of each roi, max of the pixels.

    Parameters
    ----------
    tensor_spots : torch.Tensor
        The batch of spots of shape (n, h, w).

    Returns
    -------
    maxi : torch.Tensor
        The vector of the maxi pixel value of shape (n,).

    Notes
    -----
    * No verifications are performed,
    * Call this method from a ``laueimproc.classes.spot.Spot``
        or ``laueimproc.classes.diagram.Diagram`` instance.
    """
    return torch.amax(tensor_spots, axis=(1, 2))


def compute_rois_sum(tensor_spots: torch.Tensor) -> torch.Tensor:
    """Return the intensity of each roi, sum of the pixels.

    Parameters
    ----------
    tensor_spots : torch.Tensor
        The batch of spots of shape (n, h, w).

    Returns
    -------
    intensity : torch.Tensor
        The vector of the intensity of shape (n,).

    Notes
    -----
    * No verifications are performed,
    * Call this method from a ``laueimproc.classes.spot.Spot``
        or ``laueimproc.classes.diagram.Diagram`` instance.
    """
    return torch.sum(tensor_spots, axis=(1, 2))
