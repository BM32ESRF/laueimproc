#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""

import numpy as np

from .image import Image


class Spot:
    """A single task of a Laue diagram image.

    Attributes
    ----------
    diagram : laueimproc.classes.Diagram
        The parent diagram.
    pxl_intensity : float
        The sum of the pixels value.
    roi : laueimproc.classes.image.Image
        The image of the complete spot zone (with the background removed).
    """

    def __init__(self, roi, *, _diagram=None):
        """Create a new spot with appropriated metadata.

        Parameters
        ----------
        roi : arraylike
            The region of interest.
        diagram : laueimproc.classes.diagram.Diagram, optional
            If it is provide, it corresponds to the Laue diagram that contains this spot.
            The informations linked to the diagram are recorded into the diagram instance, not here.
        """

        self._slices: tuple[int, int, int] = None  # the view position

        if not isinstance(roi, Image):
            roi = Image(roi)

        if _diagram is None:
            from .diagram import Diagram # pylint: disable=C0415
            self._diagram = Diagram._from_spot(self, roi)
            self._slices = (0, slice(0, roi.shape[0]), slice(0, roi.shape[1]))
        else:
            self._diagram = _diagram
            self._slices = roi.context

    @property
    def diagram(self):
        """Return the parent diagram."""
        return self._diagram

    @property
    def roi(self) -> Image:
        """Return the image of the region of interest."""
        index, width, height = self._slices
        return self._diagram._tensor_spots[index, :width, :height]  # pylint: disable=W0212

    @property
    def pxl_intensity(self) -> float:
        """Return the sum of the pixels value."""
        return self.sum().item()
