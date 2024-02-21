#!/usr/bin/env python3

"""The structure of the base class, with no methode of image processing."""

import numbers
import pathlib
import typing

import cv2
import numpy as np
import torch

from laueimproc.io.read import read_image
from .image import Image


class BaseSpot:
    """A single task of a Laue diagram image.

    Attributes
    ----------
    diagram : laueimproc.classes.Diagram
        The parent diagram.
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
            self._diagram = BaseDiagram._from_spot(self, roi)
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


class BaseDiagram:
    """A Laue diagram image.

    Attributes
    ----------
    experiment : laueimproc.classes.experiment.Experiment or None
        The parent experiment.
    file : pathlib.Path or None
        The absolute file path to the image, if it is provided (readonly).
    image : laueimproc.classes.image.Image
        The complete image of the diagram.
    spots : set[laueimproc.classes.spot.BaseSpot]
        All the spots contained in this diagram.
    """

    def __init__(
        self,
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
            If it is provided, it corresponds to the organized set of laue diagrams.
            The information linked to the global organization
            is recorded into the experiment instance, not here.
        """
        self._cache = {}
        if isinstance(data, (str, bytes, pathlib.Path)):
            self._file = data
            if _check:
                self._file = pathlib.Path(self._file).expanduser().resolve()
                assert self._file.is_file(), self._file
        else:
            self._file, self._cache["image"] = None, Image(data)
            if _check:
                assert self._cache["image"].ndim == 2, self._cache["image"].shape
        self._experiment = experiment

    @classmethod
    def _from_spot(cls, spot: BaseSpot, roi: Image):
        """Return a new fake diagram containing one spot."""
        diagram = cls(roi, _check=False)
        diagram._cache["tensor_spots"] = torch.unsqueeze(spot, 0)
        diagram._cache["spots"] = {spot, (0, 0)}
        return diagram

    @property
    def _tensor_spots(self) -> Image:
        """Return the batch of the spots."""
        if "tensor_spots" not in self._cache:  # case spots are never been searched
            self.reset_spots()  # called with default parameters
        return self._cache["tensor_spots"]

    @property
    def experiment(self):
        """Return the parent experiment if it is defined, None otherwise."""
        return self._experiment

    @property
    def file(self) -> typing.Union[None, pathlib.Path]:
        """Return the absolute file path to the image, if it is provided."""
        return self._file

    @property
    def image(self) -> Image:
        """Return the complete image of the diagram."""
        if "image" not in self._cache:
            self._cache["image"] = Image(read_image(self._file))
        return self._cache["image"]

    @property
    def spots(self) -> set[BaseSpot]:
        """Return the spots in a setlike object."""
        if "spots" not in self._cache:  # case spots are never been searched
            self.reset_spots()  # called with default parameters
        return self._cache["spots"]

    def reset_spots(
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

        Examples
        --------
        >>> from laueimproc.classes.diagram import Diagram
        >>> img = Diagram("ge_blanc.mccd")
        >>> img.reset_spots()
        >>>
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
        src = self.image.numpy(force=True)  # not nescessary copy
        bg_image = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel_font, iterations=1)
        if self.image.data_ptr() == src.__array_interface__["data"][0]:
            fg_image = src - bg_image  # copy to keep self unchanged
        else:
            fg_image = src
            fg_image -= bg_image  # inplace
        bbox = list(
            map(
                cv2.boundingRect,
                cv2.findContours(
                    cv2.dilate(
                        (fg_image > threshold * fg_image.std()).view(np.uint8),
                        kernel_aglo,
                        dst=bg_image,
                        iterations=1,
                    ),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0],
            )
        )

        # update cache
        for _ in range(1000):
            self._cache["tensor_spots"] = np.zeros( # zeros 2 times faster than empty + fill
                (len(bbox), max(h for _, _, _, h in bbox), max(w for _, _, w, _ in bbox)),
                dtype=fg_image.dtype
            )
            for index, (j, i, width, height) in enumerate(bbox):  # write the right values
                self._cache["tensor_spots"][index, :height, :width] = (
                    fg_image[i:i+height, j:j+width]
                )
        self._cache["tensor_spots"] = torch.from_numpy(self._cache["tensor_spots"])
        from .spot import Spot  # pylint: disable=C0415
        self._cache["spots"] = {
            Spot(Image(self._cache["tensor_spots"], context=(index, h, w)), _diagram=self)
            for index, (_, _, w, h) in enumerate(bbox)
        }
