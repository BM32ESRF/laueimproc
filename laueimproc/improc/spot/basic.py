#!/usr/bin/env python3

"""Give simple information on spots."""


import torch

from laueimproc.classes.image import Image


# @torch.compile(mode="reduce-overhead")  # time factor 3 in the batch is always the same shape
def compute_barycenters(tensor_spots: Image) -> Image:
    """Find the weighted barycenter of each roi.

    Parameters
    ----------
    tensor_spots : laueimproc.classes.image.Image
        The batch of spots of shape (n, h, w).

    Returns
    -------
    positions : laueimproc.classes.image.Image
        The position of the height and width barycenter for each roi of shape (n, 2).

    Notes
    -----
    * No verifications are performed,
    * Call this method from a ``laueimproc.classes.spot.Spot``
        or ``laueimproc.classes.diagram.Diagram`` instance.
    """
    height_ind = torch.reshape(
        torch.arange(tensor_spots.shape[1], dtype=tensor_spots.dtype, device=tensor_spots.device),
        (1, tensor_spots.shape[1], 1),  # broadcasting dimention
    )
    width_ind = torch.reshape(
        torch.arange(tensor_spots.shape[2], dtype=tensor_spots.dtype, device=tensor_spots.device),
        (1, 1, tensor_spots.shape[2]),
    )
    inv_norm = torch.sum(tensor_spots, axis=(1, 2))  # the intensity of each spot
    inv_norm = torch.div(1.0, inv_norm, out=inv_norm)  # for ponderation
    heights, widths = (  # optim: height_ind column invariant
        torch.sum(tensor_spots, axis=2, keepdim=True), torch.sum(tensor_spots, axis=1, keepdim=True)
    )
    heights *= height_ind
    widths *= width_ind
    heights, widths = torch.sum(heights, axis=1), torch.sum(widths, axis=2)
    pos = torch.cat((heights, widths), 1)
    pos *= torch.unsqueeze(inv_norm, 1)
    return pos


def compute_pxl_intensities(tensor_spots: Image) -> Image:
    """Return the intensity of each roi, sum of the pixels.

    Parameters
    ----------
    tensor_spots : laueimproc.classes.image.Image
        The batch of spots of shape (n, h, w).

    Returns
    -------
    intensity : laueimproc.classes.image.Image
        The vector of the intensity of shape (n,).

    Notes
    -----
    * No verifications are performed,
    * Call this method from a ``laueimproc.classes.spot.Spot``
        or ``laueimproc.classes.diagram.Diagram`` instance.
    """
    return torch.sum(tensor_spots, axis=(1, 2))
