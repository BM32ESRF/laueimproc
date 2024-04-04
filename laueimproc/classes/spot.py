#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""

import torch


class Spot:
    """A single task of a Laue diagram image.

    Attributes
    ----------
    anchor : tuple[int, int]
        The position (i, j) of the corner point (0, 0) of the roi in the diagram.
    bbox : tuple[int, int, int, int]
        The concatenation of `anchor` and `shape`.
    center : tuple[int, int]
        The middle point of the spot in the diagram. If shape is odd, it is rounded down.
    diagram : laueimproc.classes.Diagram
        The parent diagram.
    rawroi : torch.Tensor
        The image of the complete spot zone (in the direct image).
    roi : torch.Tensor
        The image of the complete spot zone (or the provided rois).
    shape : tuple[int, int]
        The shape of the roi in the numpy convention (readonly).
    """

    def __init__(self, roi: torch.Tensor):
        """Initialize the spot from a patch.

        Parameters
        ----------
        roi : torch.Tensor
            A very little float32 image as a region of interest.
        """
        assert isinstance(roi, torch.Tensor), roi.__class__.__name__
        assert roi.ndim == 2, roi.shape
        assert roi.dtype == torch.float32, roi.dtype

        from .diagram import Diagram  # pylint: disable=C0415
        self._diagram = Diagram(roi)
        self._index = 0

    def __getstate__(self):
        """Make the object pickleable."""
        return (self._diagram, self._index)

    def __repr__(self) -> str:
        """Give a compact representation."""
        return f"{repr(self._diagram)}.spots[{self._index}]"

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle and called by Diagram

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.
        * Return self by ease.
        """
        self._diagram, self._index = state  # pylint: disable=W0201
        # to return self allows you to create and instanciate a Spot in the same time
        return self

    @property
    def anchor(self) -> tuple[int, int]:
        """Return the position (i, j) of the corner point (0, 0) of the roi in the diagram."""
        return self.bbox[:2]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return the concatenation of `anchor` and `shape`."""
        return tuple(self._diagram.bboxes[self._index].tolist())

    @property
    def center(self) -> tuple[int, int]:
        """Return the middle point of the spot in the diagram."""
        return (self.bbox[0] + self.bbox[2]//2, self.bbox[1] + self.bbox[3]//2)

    @property
    def diagram(self):
        """Return the parent diagram."""
        return self._diagram

    @property
    def rawroi(self) -> torch.Tensor:
        """Return the patch of the brut image of the region of interest."""
        anch_i, anch_j, height, width = self.bbox
        return self._diagram.image[anch_i:anch_i+height, anch_j:anch_j+width]

    @property
    def roi(self) -> torch.Tensor:
        """Return the patch of the filtered image of the region of interest."""
        _, _, height, width = self.bbox
        return self._diagram.rois[self._index, :height, :width]  # share underground data

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the roi in the numpy convention."""
        return self.bbox[2:]
