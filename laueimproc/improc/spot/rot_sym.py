#!/usr/bin/env python3

"""Test the similarity by rotation."""

import torch
import torchvision.transforms.v2

from laueimproc.classes.tensor import Tensor
from .basic import compute_barycenters


def compute_rot_similarity(tensor_spots: Tensor) -> Tensor:
    """Search the similarity by rotation of each roi.

    Parameters
    ----------
    tensor_spots : laueimproc.classes.tensor.Tensor
        The batch of spots of shape (n, h, w).

    Returns
    -------
    similarity : laueimproc.classes.tensor.Tensor
        The similarity of each roi by rotation (n,).
    """
    # normalization
    cr_rois = tensor_spots.clone()
    cr_rois -= torch.mean(cr_rois, dim=(1, 2), keepdim=True)
    power = torch.mean(cr_rois**2, dim=(1, 2), keepdim=True)
    cr_rois *= torch.rsqrt(power, out=power)

    # preparation
    centers = (compute_barycenters(tensor_spots)-0.5).tolist()
    all_corr = torch.empty((11, cr_rois.shape[0]), dtype=cr_rois.dtype, device=cr_rois.device)

    # apply rotations
    for i, angle in enumerate(range(30, 360, 30)):
        corr = torch.cat([
            torchvision.transforms.v2.functional.rotate(
                roi.unsqueeze(0), angle,
                interpolation=torchvision.transforms.v2.InterpolationMode.BILINEAR,
                expand=False,
                center=(x, y),
                fill=0.0,
            )
            for roi, (y, x) in zip(cr_rois, centers)
        ], axis=0)
        corr *= cr_rois
        all_corr[i, :] = torch.mean(corr, dim=(1, 2))

    # compute metric
    sim = torch.min(all_corr, dim=0).values
    sim = torch.clamp(sim, min=0, out=sim)  # [-1, 1] -> [0, 1]
    return sim
