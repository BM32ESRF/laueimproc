#!/usr/bin/env python3

"""Give simple information on spots."""

import torch


# @torch.compile(mode="reduce-overhead")  # time factor 3 in the batch is always the same shape
def compute_barycenters(tensor_spots: torch.Tensor) -> torch.Tensor:
    """Find the weighted barycenter of each roi.

    Parameters
    ----------
    tensor_spots : torch.Tensor
        The batch of spots of shape (..., h, w).

    Returns
    -------
    positions : torch.Tensor
        The position of the height and width barycenter for each roi of shape (..., 2).

    Notes
    -----
    * No verifications are performed.
    * Call this method from a ``laueimproc.classes.spot.Spot``
        or ``laueimproc.classes.diagram.Diagram`` instance.
    """
    *batch_shape, height, width = tensor_spots.shape
    tensor_spots = tensor_spots.reshape(-1, height, width)  # (n, h, w)

    points_i, points_j = torch.meshgrid(
        torch.arange(
            0.5, tensor_spots.shape[-2]+0.5, dtype=tensor_spots.dtype, device=tensor_spots.device
        ),
        torch.arange(
            0.5, tensor_spots.shape[-1]+0.5, dtype=tensor_spots.dtype, device=tensor_spots.device
        ),
        indexing="ij",
    )  # (h*w,)
    points = torch.cat([points_i.ravel().unsqueeze(0), points_j.ravel().unsqueeze(0)])  # (2, h*w)
    del points_i, points_j
    points = points.unsqueeze(0)  # (1, 2, h*w)

    weight = tensor_spots.reshape(-1, 1, height*width)  # (n, 1, h*w)
    pond_points = points * weight  # (n, 2, h*w)
    pond_points /= torch.sum(weight, axis=-1, keepdim=True)  # (n, 2, h*w)
    mean = torch.sum(pond_points, axis=-1)  # (n, 2)
    del pond_points
    mean = mean.reshape(*batch_shape, 2)  # (..., 2)

    return mean

    # height_ind = torch.reshape(
    #     torch.arange(tensor_spots.shape[1], dtype=tensor_spots.dtype, device=tensor_spots.device),
    #     (1, tensor_spots.shape[1], 1),  # broadcasting dimention
    # )
    # width_ind = torch.reshape(
    #     torch.arange(tensor_spots.shape[2], dtype=tensor_spots.dtype, device=tensor_spots.device),
    #     (1, 1, tensor_spots.shape[2]),
    # )
    # inv_norm = torch.sum(tensor_spots, axis=(1, 2))  # the intensity of each spot
    # inv_norm = torch.div(1.0, inv_norm, out=inv_norm)  # for ponderation
    # heights, widths = (  # optim: height_ind column invariant
    #     torch.sum(tensor_spots, axis=2, keepdim=True), torch.sum(tensor_spots, axis=1, keepdim=True)
    # )
    # heights *= height_ind
    # widths *= width_ind
    # heights, widths = torch.sum(heights, axis=1), torch.sum(widths, axis=2)
    # pos = torch.cat((heights, widths), 1)
    # pos *= torch.unsqueeze(inv_norm, 1)
    # pos += .5  # corner of pixel to center of pixel
    # return pos


def compute_pxl_max(tensor_spots: torch.Tensor) -> torch.Tensor:
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


def compute_pxl_intensities(tensor_spots: torch.Tensor) -> torch.Tensor:
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
