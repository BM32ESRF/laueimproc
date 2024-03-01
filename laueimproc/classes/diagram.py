#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.opti.cache import auto_cache
from laueimproc.opti.parallel import auto_parallel
from .base_diagram import BaseDiagram
from .tensor import Tensor


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache
    @auto_parallel  # automaticaly multithreading
    def compute_barycenters(self) -> Tensor:
        """Compute the barycenter of each spots."""
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        barycenters = compute_barycenters(self.rois)  # relative to each spots
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)  # absolute
        return barycenters

    @auto_cache  # put the result in thread safe cache
    @auto_parallel  # automaticaly multithreading
    def compute_pxl_intensities(self) -> Tensor:
        """Compute the total pixel intensity for each spots."""
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        return compute_pxl_intensities(self.rois)
