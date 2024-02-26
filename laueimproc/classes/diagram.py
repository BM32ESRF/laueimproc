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
    def get_barycenters(self) -> Tensor:
        """Compute the barycenter of each spots."""
        barycenters = compute_barycenters(self.rois)
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)
        return barycenters

    # @BaseDiagram.spot_property("barycenter")
    # def get_barycenters(self, *, _return_tensor : bool = False) -> dict[Spot, tuple[float, float]]:
    #     """Compute the barycenter of each spots."""
    #     if "barycenters" not in self._cache:
    #         barycenters = compute_barycenters(self._tensor_spots)  # relative barycenters
    #         barycenters += self.get_anchors(_return_tensor=True)  # absolute position
    #         self._cache["barycenters"] = barycenters
    #     if _return_tensor:
    #         return self._cache["barycenters"]
    #     return {
    #         spot: (
    #             self._cache["barycenters"][index, 0].item(),
    #             self._cache["barycenters"][index, 1].item(),
    #         )
    #         for index, spot in enumerate(self.spots)
    #     }

    # @BaseDiagram.spot_property("pxl_intensity")
    # def get_pxl_intensities(self, *, _return_tensor : bool = False) -> dict[Spot, float]:
    #     """Compute the total pixel intensity for each spots."""
    #     if "pxl_intensities" not in self._cache:
    #         self._cache["pxl_intensities"] = compute_pxl_intensities(self._tensor_spots)
    #     if _return_tensor:
    #         return self._cache["pxl_intensities"]
    #     return {
    #         spot: self._cache["pxl_intensities"][index].item()
    #         for index, spot in enumerate(self.spots)
    #     }
