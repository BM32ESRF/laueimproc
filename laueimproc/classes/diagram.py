#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import typing

import numpy as np

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from .base_diagram import BaseDiagram
from .tensor import Tensor


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @BaseDiagram.enable_caching  # put the result in cache
    def compute_barycenters(self) -> Tensor:
        """Compute the barycenter of each spots."""
        barycenters = compute_barycenters(self.rois)  # relative to each spots
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)  # absolute
        return barycenters

    @BaseDiagram.enable_caching  # put the result in cache
    def compute_pxl_intensities(self) -> Tensor:
        """Compute the total pixel intensity for each spots."""
        return compute_pxl_intensities(self.rois)
