#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""


from .tensor import Tensor



class Spot:
    """A single task of a Laue diagram image.

    Attributes
    ----------
    anchor : tuple[int, int]
        The position (i, j) of the corner point (0, 0) of the roi in the diagram.
    bbox : tuple[int, int, int, int]
        The concatenation of `anchor` and `shape`.
    diagram : laueimproc.classes.Diagram
        The parent diagram.
    roi : laueimproc.classes.tensor.Tensor
        The image of the complete spot zone (or the provided rois).
    roi_brut : laueimproc.classes.tensor.Tensor
        The image of the complete spot zone (in the direct image).
    shape : tuple[int, int]
        The shape of the roi in the numpy convention (readonly).
    """

    def __getstate__(self, cache: bool = False):
        """Make the object pickleable."""
        if cache:
            return (self._diagram, self._index, self._bbox, self._cache)
        return (self._diagram, self._index, self._bbox)

    def __repr__(self) -> str:
        """Give a compact representation."""
        return f"{self.__class__.__name__}({', '.join(map(str, self._bbox))})"

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle and called by Diagram

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.
        * Return self by ease.
        """
        # declaration
        from .diagram import Diagram  # pylint: disable=C0415
        # the region of interest anchors (i, j, h, w)
        self._bbox: tuple[int, int, int, int]  # pylint: disable=W0201
        # contains the optional cached data
        self._cache: dict[str] = {}  # pylint: disable=W0201
        # the parent diagram that contains this spot (cyclic reference)
        self._diagram: Diagram  # pylint: disable=W0201
        # the position in the diagram (self is self._diagram.spots[self._index])
        self._index: int  # pylint: disable=W0201

        # fill the attributes
        self._diagram, self._index, self._bbox = state[:3]  # pylint: disable=W0201
        if len(state) == 4:
            self._cache = state[3]  # pylint: disable=W0201

        # to return self allows you to create and instanciate a Spot in the same time
        return self

    # def __getattr__(self, attr):
    #     """Link automaticaly the idiagram spot properties as a spot attribute."""
    #     def obj_meth(*args, **kwargs):
    #         props = meth(self.diagram, *args, **kwargs)
    #         if not isinstance(props, dict):
    #             raise ValueError(
    #                 f"the method {meth} must return a dictionary with a spot key, not {props}"
    #             )
    #         if self not in props:
    #             raise ValueError(
    #                 f"the spot {self} must be a key of the returned dictionary of the meth {meth}"
    #             )
    #         return props[self]

    #     if (meth := self.diagram._spot_property.get(attr, None)) is not None:
    #         return obj_meth()
    #     if (
    #         attr.startswith("get_")
    #         and (meth := self.diagram._spot_property.get(attr[4:], None)) is not None
    #     ):
    #         return obj_meth
    #     raise AttributeError(f"{self} has not attribute {attr}")

    @property
    def anchor(self):
        """Return the position (i, j) of the corner point (0, 0) of the roi in the diagram."""
        return self._bbox[:2]

    @property
    def bbox(self):
        """Return the concatenation of `anchor` and `shape`."""
        return self._bbox

    @property
    def diagram(self):
        """Return the parent diagram."""
        return self._diagram

    @property
    def roi(self) -> Tensor:
        """Return the patch of the filtered image of the region of interest."""
        _, _, height, width = self._bbox
        return self._diagram.rois[self._index, :height, :width]  # share underground data

    @property
    def roi_brut(self) -> Tensor:
        """Return the patch of the brut image of the region of interest."""
        anch_i, anch_j, height, width = self._bbox
        return self._diagram.image[anch_i, anch_j, :height, :width]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the roi in the numpy convention."""
        return self._bbox[2:]
