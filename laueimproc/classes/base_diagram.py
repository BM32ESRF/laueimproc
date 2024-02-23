#!/usr/bin/env python3

"""Define the pytonic structure of a basic Diagram."""

import inspect
import numbers
import pathlib
import pickle
import typing
import warnings

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from laueimproc.io.read import read_image
from laueimproc.improc.peaks_search import peaks_search
from .image import Image
from .spot import Spot



class BaseDiagram:
    """A Basic diagram with the fondamental structure.

    Attributes
    ----------
    file : pathlib.Path or None
        The absolute file path to the image if provided, None otherwise (readonly).
    bboxes : np.ndarray
        The tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots of shape (n, 4) (readonly).
    image : laueimproc.classes.image.Image
        The complete brut image of the diagram (readonly).
    rois : laueimproc.classes.image.Image
        The tensor of the regions of interest for each spots (readonly).
        For writing, use `self.spots = ...`.
    spots : list[laueimproc.classes.spot.Spot]
        All the spots contained in this diagram (read and write).
    """

    _spot_property: dict[str, callable] = {}  # to each property name, associate the method

    def __init__(
        self,
        data: typing.Union[np.ndarray, torch.Tensor, str, bytes, pathlib.Path],
        *, _check: bool = True,
    ):
        """Create a new diagram with appropriated metadata.

        Parameters
        ----------
        data : path or arraylike
            The filename or the array/tensor use as a diagram.
            For memory management, it is better to provide a pathlike rather than an array.
        """
        self._cache: dict[str] = {}  # contains the optional cached data
        self._file_or_data: typing.Union[pathlib.Path, Image]  # the path to the image file
        self._history: list[str] = []  # the history of the actions performed
        self._spots: typing.Optional[list[Spot]] = None # the spots of the diagram

        if isinstance(data, (str, bytes, pathlib.Path)):
            self._file_or_data = data
            if _check:
                self._file_or_data = pathlib.Path(self._file_or_data).expanduser().resolve()
                assert self._file_or_data.is_file(), self._file_or_data
        else:
            warnings.warn("please provide a path rather than an array", ResourceWarning)
            self._file_or_data = Image(data)
            assert self._file_or_data.ndim == 2, self._file_or_data.shape

    def __getstate__(self, cache: bool = False):
        """Make the object pickleable."""
        if self._spots is None:
            spots_no_diagram = None
        else:
            spots_no_diagram = [
                Spot.__new__(Spot).__setstate__(s.__getstate__(cache=cache))
                for s in self._spots
            ]
            for spot in spots_no_diagram:
                spot._diagram = None  # in order to avoid cyclic reference
        if cache:
            return (self._file_or_data, self._history, spots_no_diagram, self._cache)
        return (self._file_or_data, self._history, spots_no_diagram)

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle.

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.
        * Return self by ease.
        """
        self._file_or_data, self._history, self._spots = state[:3]
        if self._spots is not None:
            for spot in self._spots:
                spot._diagram = self
        if len(state) == 4:
            self._cache = state[3]

    @property
    def bboxes(self) -> np.ndarray[int]:
        """Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width)."""
        if "bboxes" not in self._cache:
            self._cache["bboxes"] = np.array([s.bbox for s in self.spots], dtype=int)
        return self._cache["bboxes"]

    def clone(self, deep: bool = True, cache: bool = True):
        """Instanciate a new identical diagram.

        Parameters
        ----------
        deep : boolean, default=True
            If True, the memory of the new created diagram object is totally
            independ of this object (slow but safe). Otherwise, `image` and `rois` attributes
            and cached big data share the same memory view (Image). So modifying one of these
            attributes in one diagram will modify the same attribute in the other diagram.
            It if realy faster but not safe.
        cache : boolean, default=True
            Copy the cache into the new diagram if True (default), or live the cache empty if False.
        """
        assert isinstance(deep, bool), deep.__class__.__name__
        assert isinstance(cache, bool), cache.__class__.__name__

        state = self.__getstate__(cache=cache)
        if deep:
            state = pickle.loads(pickle.dumps(state))
        new_diagram = self.__class__.__new__(self.__class__)  # create a new diagram
        new_diagram.__setstate__(state)  # initialise (fill) the new diagram
        return new_diagram

    # def get_anchors(self, *, _return_tensor : bool = False) -> dict[Spot, tuple[float, float]]:
    #     """Return the i and j position for each spot roi anchor."""
    #     if "anchors" not in self._cache:  # case spots are never been searched
    #         self.peaks_search()  # called with default parameters
    #     if _return_tensor:
    #         return self._cache["anchors"]
    #     return {
    #         spot: (
    #             self._cache["anchors"][index, 0].item(),
    #             self._cache["anchors"][index, 1].item(),
    #         )
    #         for index, spot in enumerate(self.spots)
    #     }

    @property
    def file(self) -> typing.Union[None, pathlib.Path]:
        """Return the absolute file path to the image, if it is provided."""
        if isinstance(self._file_or_data, pathlib.Path):
            return self._file_or_data
        return None

    def get_spots(self) -> list[Spot]:
        """Alias to the `spot` attribute."""
        warnings.warn("call `self.spots` rather than `self.get_spots()`", DeprecationWarning)
        return self.spots

    @property
    def image(self) -> Image:
        """Return the complete image of the diagram."""
        if "image" not in self._cache:
            if isinstance(self._file_or_data, pathlib.Path):
                self._cache["image"] = Image(read_image(self._file_or_data))
            else:
                self._cache["image"] = self._file_or_data
        return self._cache["image"]

    def plot(self, fig: Figure) -> None:
        """Prepare for display the diagram and the spots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The matplotlib figure to complete.

        Notes
        -----
        It doesn't create the figure and call show.
        Use `self.show()` to Display the diagram from scratch.
        """
        assert isinstance(fig, Figure), fig.__class__.__name__
        if isinstance(self._file_or_data, pathlib.Path):
            fig.suptitle(f"Diagram {self._file_or_data.name}")
        else:
            fig.suptitle(f"Diagram from Image of id {id(self._file_or_data)}")
        axes = fig.add_subplot()
        axes.imshow(self.image, aspect="equal", interpolation="bicubic", norm="log", cmap="gray")
        if self._spots:
            axes.plot(
                np.vstack((
                    self.bboxes[:, 1],
                    self.bboxes[:, 1],
                    self.bboxes[:, 1]+self.bboxes[:, 3],
                    self.bboxes[:, 1]+self.bboxes[:, 3],
                    self.bboxes[:, 1],
                )),
                np.vstack((
                    self.bboxes[:, 0],
                    self.bboxes[:, 0]+self.bboxes[:, 2],
                    self.bboxes[:, 0]+self.bboxes[:, 2],
                    self.bboxes[:, 0],
                    self.bboxes[:, 0],
                )),
                color="blue",
                scalex=False,
                scaley=False,
            )
            # axes.scatter(self.bboxes[:, 1], self.bboxes[:, 0], color="blue")


    @property
    def rois(self) -> Image:
        """Return the tensor of the rois of the spots."""
        if "rois" not in self._cache:
            raise NotImplementedError
        return self._cache["rois"]

    def show(self) -> None:
        """Display the diagram, ccreate the matplotlib context of `self.plot`."""
        fig = plt.Figure(layout="tight")
        self.plot(fig)
        plt.show()

    # @classmethod
    # def spot_property(cls, name: str) -> callable:
    #     """Decorate a method to include it in the filters.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the property.
    #         It became the name of the spot attribute.
    #     """
    #     assert isinstance(name, str), name.__class__.__name__
    #     assert name, "the property name can not be ''"
    #     def meth_decorator(meth):
    #         assert name not in cls._spot_property, \
    #             f"the {name} property of {meth} is already defined for {cls._spot_property[name]}"
    #         assert "_return_tensor" in inspect.signature(meth).parameters, \
    #             f"the method {meth} requiere the bool parameter '_return_tensor'"
    #         ret_param = inspect.signature(meth).parameters["_return_tensor"]
    #         assert ret_param.kind == ret_param.KEYWORD_ONLY, \
    #             f"the parameter '_return_tensor' of the method {meth} has to be keyword only"
    #         cls._spot_property[name] = meth
    #         return meth
    #     return meth_decorator

    def searched_spots(self, **kwargs) -> None:
        """Search all the spots in this diagram, store the result into the `spots` attribute.

        Parameters
        ----------
        kwargs : dict
            Transmitted to ``laueimproc.improc.peaks_search.peaks_search``.
        """
        # peaks search
        image = self.image
        rois, bboxes = peaks_search(image, **kwargs)

        # cast into spots objects
        self._cache = {"image": image, "rois": rois}  # reset cache to prevent wrong data
        self._spots = [
            Spot.__new__(Spot).__setstate__((self, index, bbox))
            for index, bbox in enumerate(bboxes.tolist())
        ]

    @property
    def spots(self) -> list[Spot]:
        """Return the spots in a unordered set container."""
        if self._spots is None:  # case spots are never been searched
            warnings.warn(
                (
                    "please, initialize the spots before accessing the `.spots` attribute, "
                    "therefore, the `searched_spots` method is automatically invoked "
                    "with the default parameters"
                ),
                RuntimeWarning,
            )
            self.searched_spots()  # called with default parameters
        return self._spots.copy()  # copy is slow but it is a strong protection against the user

    # @spots.setter
    # def spots(self, new_spots: typing.Container[Spot]):
    #     """Set the new spots as the current spots."""
    #     if not isinstance(new_spots, list):
    #         assert hasattr(new_spots, "__iter__"), new_spots.__class__.__name__
    #         new_spots = list(new_spots)
    #     assert all(isinstance(s, Spot) for s in new_spots), new_spots
    #     assert all(s.diagram is self for s in new_spots), "all spots must come from this diagram"

    #     # prepare new spots
    #     new_cache = {}
    #     if not new_spots:
    #         new_cache["tensor_spots"] = Image(torch.empty((0, 1, 1), dtype=torch.float32))
    #         new_cache["anchors"] = Image(torch.empty((0, 2), dtype=torch.float32))
    #     else:
    #         new_cache["anchors"] = self._cache["anchors"][[s._slices[0] for s in new_spots]]
    #         new_cache["tensor_spots"] = Image(torch.zeros(
    #             (
    #                 len(new_spots),
    #                 max(s.shape[0] for s in new_spots),
    #                 max(s.shape[1] for s in new_spots),
    #             ),
    #             dtype=new_spots[0].roi.dtype,
    #             device=new_spots[0].roi.device,
    #         ))
    #         for index, spot in enumerate(new_spots):
    #             new_cache["tensor_spots"][index, :spot.shape[0], :spot.shape[1]] = spot.roi
    #             spot._slices = (index, *spot.shape)
    #     new_cache["spots"] = new_spots

    #     # set new spots, rest cache
    #     if "image" in self._cache:
    #         new_cache["image"] = self._cache["image"]
    #     self._cache = new_cache

    # def estimate_background(
    #     self, kernel_font: typing.Optional[np.ndarray[np.uint8, np.uint8]] = None
    # ) -> Image:
    #     """Return the background of the brut image``.

    #     Parameters
    #     ----------
    #     kernel_font : np.ndarray[np.uint8, np.uint8], optional
    #         Transitted to ``laueimproc.improc.peaks_search.estimate_background``.

    #     Returns
    #     -------
    #     background : Image
    #         An estimation of the background.
    #     """
    #     if "background" not in self._cache:
    #         self._cache["background"] = Image(
    #             estimate_background(self.image.numpy(force=True), kernel_font)
    #         )
    #     return self._cache["background"]
