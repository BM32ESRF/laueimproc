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


def compute_rois_centroid(
    data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False
) -> torch.Tensor:
    r"""Find the weighted barycenter of each roi.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
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
    >>> from laueimproc.improc.spot.basic import compute_rois_centroid
    >>> patches = [
    ...     torch.tensor([[0.5]]),
    ...     torch.tensor([[0.5, 0.5], [0.5, 1.0]]),
    ...     torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]]),
    ... ] * 2
    >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes())
    >>> bboxes = torch.tensor([[ 0,  0,  1,  1],
    ...                        [ 0,  0,  2,  2],
    ...                        [ 0,  0,  1,  5],
    ...                        [10, 10,  1,  1],
    ...                        [10, 10,  2,  2],
    ...                        [10, 10,  1,  5]], dtype=torch.int16)
    >>> print(compute_rois_centroid(data, bboxes))
    tensor([[ 0.5000,  0.5000],
            [ 1.1000,  1.1000],
            [ 0.5000,  2.5000],
            [10.5000, 10.5000],
            [11.1000, 11.1000],
            [10.5000, 12.5000]])
    >>> torch.allclose(
    ...     compute_rois_centroid(data, bboxes),
    ...     compute_rois_centroid(data, bboxes, _no_c=True),
    ... )
    True
    >>>
    """
    if not _no_c and c_basic is not None:
        return torch.from_numpy(
            c_basic.compute_rois_centroid(data, bboxes.numpy(force=True))
        ).to(bboxes.device)

    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)  # (n, h, w), more verif here
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
    pond_points /= torch.sum(weight, axis=2, keepdim=True)  # (n, 2, h*w)
    mean = torch.sum(pond_points, axis=2)  # (n, 2)
    mean += bboxes[:, :2].to(rois.dtype)  # relative to absolute base
    return mean


def compute_rois_max(data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False) -> torch.Tensor:
    r"""Return the argmax of the intensity and the intensity max of each roi.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.

    Returns
    -------
    imax_pos1_pos2 : torch.Tensor
        The concatenation of the colum vectors
        of the argmax along i axis, argmax along j axis and the max intensity.
        The shape is (n, 3).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.spot.basic import compute_rois_max
    >>> patches = [
    ...     torch.tensor([[0.5]]),
    ...     torch.tensor([[0.5, 0.5], [0.5, 1.0]]),
    ...     torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]]),
    ... ] * 2
    >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes())
    >>> bboxes = torch.tensor([[ 0,  0,  1,  1],
    ...                        [ 0,  0,  2,  2],
    ...                        [ 0,  0,  1,  5],
    ...                        [10, 10,  1,  1],
    ...                        [10, 10,  2,  2],
    ...                        [10, 10,  1,  5]], dtype=torch.int16)
    >>> print(compute_rois_max(data, bboxes))
    tensor([[ 0.5000,  0.5000,  0.5000],
            [ 1.5000,  1.5000,  1.0000],
            [ 0.5000,  2.5000,  0.5000],
            [10.5000, 10.5000,  0.5000],
            [11.5000, 11.5000,  1.0000],
            [10.5000, 12.5000,  0.5000]])
    >>> torch.allclose(
    ...     compute_rois_max(data, bboxes),
    ...     compute_rois_max(data, bboxes, _no_c=True),
    ... )
    True
    >>>
    """
    if not _no_c and c_basic is not None:
        return torch.from_numpy(
            c_basic.compute_rois_max(data, bboxes.numpy(force=True))
        ).to(bboxes.device)

    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)  # (n, h, w), more verif here
    nb_spots, height, width = rois.shape
    out = torch.empty((nb_spots, 3), dtype=rois.dtype, device=rois.device)
    out[:, :2] = bboxes[:, :2] + 0.5
    out[:, 2], indice = torch.max(rois.reshape(nb_spots, height*width), axis=1)
    out[:, 0] += indice // width
    out[:, 1] += indice % width
    return out


def compute_rois_sum(data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False) -> torch.Tensor:
    r"""Return the intensity of each roi, sum of the pixels.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.

    Returns
    -------
    intensity : torch.Tensor
        The vector of the intensity of shape (n,).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.spot.basic import compute_rois_sum
    >>> patches = [
    ...     torch.tensor([[0.5]]),
    ...     torch.tensor([[0.5, 0.5], [0.5, 1.0]]),
    ...     torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]]),
    ... ]
    >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes())
    >>> bboxes = torch.tensor([[ 0,  0,  1,  1],
    ...                        [ 0,  0,  2,  2],
    ...                        [ 0,  0,  1,  5]], dtype=torch.int16)
    >>> print(compute_rois_sum(data, bboxes))
    tensor([0.5000, 2.5000, 0.5000])
    >>> torch.allclose(
    ...     compute_rois_sum(data, bboxes),
    ...     compute_rois_sum(data, bboxes, _no_c=True),
    ... )
    True
    >>>
    """
    if not _no_c and c_basic is not None:
        return torch.from_numpy(
            c_basic.compute_rois_sum(data, bboxes.numpy(force=True))
        ).to(bboxes.device)

    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)  # (n, h, w), more verif here
    return torch.sum(rois, axis=(1, 2))
