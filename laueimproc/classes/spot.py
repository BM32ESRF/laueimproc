#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""

from .base import BaseSpot


class Spot(BaseSpot):
    """A single task of a Laue diagram image.

    Attributes
    ----------
    pxl_intensity : float
        The sum of the pixels value.
    """

    def __init__(self, roi, *, _diagram=None):
        """Update the BaseSpot by a simple Spot.

        Parameters
        ----------
        roi : arraylike
            Transmitted to ``laueimproc.classes.base.BaseSpot``.
        diagram : laueimproc.classes.diagram.Diagram, optional
            Transmitted to ``laueimproc.classes.base.BaseSpot``.
        """
        from .diagram import Diagram  # pylint: disable=C0415
        super().__init__(roi, _diagram=_diagram)
        new_diagram = self._diagram.__new__(Diagram)
        new_diagram.__dict__.update(self._diagram.__dict__)
        self._diagram = new_diagram

    @property
    def pxl_intensity(self) -> float:
        """Return the sum of the pixels value."""
        return self.roi.sum().item()
