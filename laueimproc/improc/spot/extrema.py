#!/usr/bin/env python3

"""Search the number of extremums in each rois."""

import logging

import torch

try:
    from laueimproc.improc.spot import c_extrema
except ImportError:
    logging.warning(
        "failed to import laueimproc.improc.spot.c_extrema, a slow python version is used instead"
    )
    c_extrema = None
from laueimproc.opti.rois import rawshapes2rois


KERNEL_STRICT = torch.tensor([[[[0., -1., 0.], [0., 1., 0.], [0., 0., 0.]]],
                              [[[-1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]]])
KERNEL_LARGE = torch.tensor([[[[0., 0., -1.], [0., 1., 0.], [0., 0., 0.]]],
                             [[[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]]],
                             [[[0., 0., 0.], [0., 1., 0.], [0., 0., -1.]]],
                             [[[0., 0., 0.], [0., 1., 0.], [0., -1., 0.]]]])


def compute_rois_nb_peaks(
    data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False
) -> torch.Tensor:
    r"""Find the number of extremums in each roi.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.

    Returns
    -------
    nbr : torch.Tensor
        The number of extremum in each roi of shape (n,) and dtype int16.
        The detection is based on canny filtering.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.spot.extrema import compute_rois_nb_peaks
    >>> patches = [
    ...     torch.tensor([[.5]]),
    ...     torch.tensor([[.5, .5], [.5, .5]]),
    ...     torch.tensor([[.5, .5, .5], [.5, .5, .5], [.5, .5, .5]]),
    ...     torch.tensor([[.1, .0, .1, .2, .3, .3, .2, .1, .1],
    ...                   [.0, .2, .1, .2, .3, .2, .1, .1, .1],
    ...                   [.0, .1, .2, .2, .2, .1, .1, .0, .0],
    ...                   [.1, .0, .1, .1, .2, .1, .0, .0, .0],
    ...                   [.2, .1, .2, .3, .4, .6, 4., .3, .1],
    ...                   [.3, .2, .3, .5, .8, .8, .8, .6, .3],
    ...                   [.4, .3, .5, .8, .8, .8, .8, .7, .3],
    ...                   [.2, .3, .5, .7, .8, .8, .7, .6, .3],
    ...                   [.2, .2, .4, .6, .7, .7, .5, .3, .2]]),
    ... ]
    >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes())
    >>> bboxes = torch.tensor([[ 0,  0,  1,  1],
    ...                        [ 0,  0,  2,  2],
    ...                        [ 0,  0,  3,  3],
    ...                        [ 0,  0,  9,  9]], dtype=torch.int16)
    >>> print(compute_rois_nb_peaks(data, bboxes))
    tensor([1, 1, 1, 5], dtype=torch.int16)
    >>> (
    ...     compute_rois_nb_peaks(data, bboxes) == compute_rois_nb_peaks(data, bboxes, _no_c=True)
    ... ).all()
    tensor(True)
    >>>
    """
    if not _no_c and c_extrema is not None:
        return torch.from_numpy(
            c_extrema.compute_rois_nb_peaks(data, bboxes.numpy(force=True))
        ).to(bboxes.device)

    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)  # (n, h, w), more verif here
    kernel_strict = KERNEL_STRICT.clone().to(rois.device)  # clone for thread safe
    kernel_large = KERNEL_LARGE.clone().to(rois.device)
    cond_1 = torch.nn.functional.conv2d(  # (n, 4, h, w)
        rois.unsqueeze(1), kernel_strict, padding=1
    ) > 0
    cond_1 = torch.all(cond_1, dim=1)  # (n, h-2, w-2)
    cond_2 = torch.nn.functional.conv2d(  # (n, 4, h, w)
        rois.unsqueeze(1), kernel_large, padding=1
    ) >= 0
    cond_2 = torch.all(cond_2, dim=1)  # (n, h-2, w-2)
    cond = torch.logical_and(cond_1, cond_2)
    nb_peaks = torch.sum(cond, dim=(1, 2), dtype=torch.int16)  # (n,)
    return nb_peaks
