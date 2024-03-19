#!/usr/bin/env python3

"""Test the similarity by rotation."""

import math

import cv2
import numpy as np
import torch

from laueimproc.improc.spot.basic import compute_barycenters


def compute_rot_sym(tensor_spots: torch.Tensor) -> torch.Tensor:
    """Search the similarity by rotation of each roi.

    Parameters
    ----------
    tensor_spots : torch.Tensor
        The batch of spots of shape (n, h, w).

    Returns
    -------
    similarity : torch.Tensor
        The similarity of each roi by rotation (n,).
    """
    # normalization
    cr_rois = tensor_spots.clone()
    cr_rois -= torch.mean(cr_rois, dim=(1, 2), keepdim=True)
    power = torch.mean(cr_rois**2, dim=(1, 2), keepdim=True)
    cr_rois *= torch.rsqrt(power, out=power)

    # preparation
    centers = compute_barycenters(tensor_spots)
    centers -= 0.5
    centers = centers.tolist()
    all_corr = torch.empty((11, cr_rois.shape[0]), dtype=cr_rois.dtype, device=cr_rois.device)
    cr_rois_np = cr_rois.numpy(force=True)
    _, height, width = cr_rois_np.shape

    # apply rotations
    for i, angle in enumerate(range(0, 360, 30)):
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dst = np.float32([(.5*width, .5*height), (.5*width+cos, .5*height+sin), (.5*width-sin, .5*height+cos)])
        corr = torch.cat([
            torch.from_numpy(cv2.warpAffine(
                roi,
                # cv2.getRotationMatrix2D((j, i), angle, 1.0),
                cv2.getAffineTransform(np.float32([(j, i), (j+1, i), (j, i+1)]), dst),
                dsize=(width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )).unsqueeze(0)
            for roi, (i, j) in zip(cr_rois_np, centers)
        ], axis=0)
        # import matplotlib.pyplot as plt
        # plt.imshow(corr[0].numpy(), cmap="gray"); plt.show()
        if i:
            corr *= cr_rois
            all_corr[i-1, :] = torch.mean(corr, dim=(1, 2))
        else:
            cr_rois = corr

    # compute metric
    sim = torch.min(all_corr, dim=0).values
    sim = torch.clamp(sim, min=0, out=sim)  # [-1, 1] -> [0, 1]
    return sim

if __name__ == '__main__':
    from laueimproc import Diagram
    from laueimproc.io.download import get_samples
    all_files = sorted(get_samples().glob("*.jp2"))
    diagrams = [Diagram(f) for f in all_files]
    for diagram in diagrams:
        diagram.find_spots()
    for diagram in diagrams:
        rois = diagram.rois
        compute_rot_sym(rois)
