#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import numbers
import pathlib
import typing

import cv2
import numpy as np
import torch

from laueimproc.io import read_image
from .image import Image
from .spot import Spot


class Diagram(Image):
    """A Laue diagram image.

    Attributes
    ----------
    experiment : laueimproc.classes.experiment.Experiment or None
        The parent experiment.
    spots : set[laueimproc.classes.spot.Spot]
        All the spots contained in this diagram.
    """

    def __new__(
        cls,
        data: typing.Union[np.ndarray, torch.Tensor, str, bytes, pathlib.Path],
        experiment=None,
        *, _check: bool = True,
    ):
        """Create a new diagram with appropriated metadata.

        Parameters
        ----------
        data : path or arraylike
            The filename or the array/tensor use as a diagram.
        experiment : Experiment, optional
            If it is provide, it corresponds to the organized set of laue diagrams.
            The information linked to the global organization
            is recorded into the experiment instance, not here.
        """
        if isinstance(data, (str, bytes, pathlib.Path)):
            data = read_image(data)
        context = {}
        if experiment is not None:
            context["experiment"] = experiment
        diagram = super().__new__(cls, data, context)
        if _check:
            assert diagram.ndim == 2, diagram.shape
        return diagram

    @property
    def experiment(self):
        """Return the parent experiment if it is defined, None otherwise."""
        return self.context.get("experiment", None)

    @property
    def spots(self) -> set[Spot]:
        """Return the spots."""
        if "spots" not in self.context:  # case spots are never been searched
            self.reset_spots()  # called with default parameters
        return self.context["spots"]

    def init_all_spots(
        self,
        threshold: typing.Optional[float] = None,
        kernel_font: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
        kernel_aglo: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None,
    ) -> None:
        """Search all the spots in this diagram, store the result into self.spots.

        Parameters
        ----------
        threshold : float, optional
            Only keep the pics having a max intensity divide by the standard deviation
            of the image without background > threshold.
            If it is not provide, it trys to use the threshold defined in ``experiment``.
        kernel_font : np.ndarray[np.uint8, np.uint8], optional
            The structurant element for the background estimation by morphological opening.
            If it is not provide, it trys to use the kernel defined in ``experiment``.
        kernel_aglo : np.ndarray[np.uint8, np.uint8], optional
            The structurant element for the aglomeration of close grain
            by morhpological dilatation applied on the thresholed image.
            If it is not provide, it trys to use the kernel defined in ``experiment``.

        Notes
        -----
        * This method doesn't have to be called because it is called with default parameters
            when we acces to the ``spots`` attribute.
        * If you wish to fine tune the pic search with specifics parameters,
            and only in this case, you should to run this method.
        """
        # get and check the parameters
        experiment = self.experiment
        if threshold is not None:
            assert isinstance(threshold, numbers.Real), threshold.__class__.__type__
            assert threshold > 0.0, threshold
            threshold = float(threshold)
        else:
            threshold = 5.1 if experiment is None else experiment.threshold
        if kernel_font is not None:
            assert isinstance(kernel_font, np.ndarray), threshold.__class__.__type__
            assert kernel_font.dtype == np.uint8, kernel_font.dtype
            assert kernel_font.ndim == 2, kernel_font.shape
            assert kernel_font.shape[0] % 2 and kernel_font.shape[1] % 2, \
                f"the kernel has to be odd, current shape is {kernel_font.shape}"
        elif experiment is not None:
            kernel_font = experiment.threshold
        else:
            kernel_font = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        if kernel_aglo is not None:
            assert isinstance(kernel_aglo, np.ndarray), threshold.__class__.__type__
            assert kernel_aglo.dtype == np.uint8, kernel_aglo.dtype
            assert kernel_aglo.ndim == 2, kernel_aglo.shape
            assert kernel_aglo.shape[0] % 2 and kernel_aglo.shape[1] % 2, \
                f"the kernel has to be odd, current shape is {kernel_aglo.shape}"
        elif experiment is not None:
            kernel_aglo = experiment.threshold
        else:
            kernel_aglo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # pic search
        src = self.numpy(force=True)  # not nescessary copy
        bg_image = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel_font, iterations=1)
        if self.data_ptr() == src.__array_interface__["data"][0]:
            fg_image = src - bg_image  # copy to keep self unchanged
        else:
            fg_image = src
            fg_image -= bg_image  # inplace
        thresh_image = (fg_image > threshold * fg_image.std()).view(np.uint8)
        dilated_image = cv2.dilate(thresh_image, kernel_aglo, dst=bg_image, iterations=1)
        outlines, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = [cv2.boundingRect(outl) for outl in outlines]

        # update context
        self.context["spots"] = [ # TODO convert in spot
            fg_image[i:i+h, j:j+w]
            for j, i, w, h in bbox
        ]
        import matplotlib.pyplot as plt
        for spot in self.context["spots"]:
            plt.imshow(spot)
            plt.show()

