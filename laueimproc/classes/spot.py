#!/usr/bin/env python3

"""Define the data sructure of a single spot in a Laue diagram image."""


from .image import Image


class Spot(Image):
    """A single task of a Laue diagram image.

    Attributes
    ----------
    diagram : laueimproc.classes.Diagram or None
        The parent diagram.
    intensity : float
        The sum of the pixels value.
    """

    def __new__(cls, data, diagram=None, *, _check: bool = True):
        """Create a new spot with appropriated metadata.

        Parameters
        ----------
        diagram : laueimproc.classes.diagram.Diagram, optional
            If it is provide, it corresponds to the Laue diagram that contains this spot.
            The informations linked to the diagram are recorded into the diagram instance, not here.
        """
        context = {}
        if diagram is not None: # importation here to avoid cyclic import
            context["diagram"] = diagram
        spot = super().__new__(cls, data, context)
        if _check:
            if diagram:
                from .diagram import Diagram # pylint: disable=C0415
                assert isinstance(diagram, Diagram), \
                    f"the diagram has to be of type Diagram, not {diagram.__class__.__name__}"
            assert spot.ndim == 2, spot.shape
        return spot

    @property
    def diagram(self):
        """Return the parent diagram if it is defined, None otherwise."""
        return self.context.get("diagram", None)

    @property
    def intensity(self) -> float:
        """Return the sum of the pixels value."""
        raise NotImplementedError
