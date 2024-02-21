#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

from .base import BaseDiagram



class Diagram(BaseDiagram):
    """A Laue diagram image.

    Attributes
    ----------
    experiment : laueimproc.classes.experiment.Experiment or None
        The parent experiment.
    file : pathlib.Path or None
        The absolute file path to the image, if it is provided (readonly).
    image : laueimproc.classes.image.Image
        The complete image of the diagram.
    spots : set[laueimproc.classes.spot.Spot]
        All the spots contained in this diagram.
    """

    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.image)
