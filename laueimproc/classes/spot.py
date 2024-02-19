#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""


from .image import Image


class Spot(Image):
    """A Laue diagram single spot image.

    Attributes
    ----------
    diagram: laueimproc.classes.Diagram or None
    """

    def __new__(cls, data, diagram=None, *, _check: bool = True):
        """Create a new spot with appropriate metadata.

        Parameters
        ----------
        diagram : laueimproc.classes.Diagram or None
            If it is provide, it correspond to the Laue diagram that contains this spot.
            The informations linked to the diagram are recorded into the diagram instance, not here.
        """
        metadata = {}
        if diagram is not None: # importation here to avoid cyclic import
            if _check:
                from .diagram import Diagram # pylint: disable=C0415
                assert isinstance(diagram, Diagram), \
                    f"diagram has to be of type Diagram, not {diagram.__class__.__name__}"
            metadata["diagram"] = diagram
        return super().__new__(cls, data, metadata)

    @property
    def diagram(self):
        """Return the parent diagram if it is define, None otherwise."""
        return self.metadata.get("diagram", None)
