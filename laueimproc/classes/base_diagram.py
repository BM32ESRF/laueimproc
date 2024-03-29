#!/usr/bin/env python3

"""Define the pytonic structure of a basic Diagram."""

import hashlib
import math
import numbers
import pathlib
import pickle
import re
import sys
import threading
import typing
import warnings

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch

from laueimproc.common import bytes2human
from laueimproc.improc.peaks_search import peaks_search
from laueimproc.io.read import read_image, to_floattensor
from laueimproc.opti.cache import auto_cache, getsizeof
from laueimproc.opti.comp import compress_rois, decompress_rois
from laueimproc.opti.manager import DiagramManager
from .spot import Spot



class BaseDiagram:
    """A Basic diagram with the fondamental structure.

    Attributes
    ----------
    bboxes : torch.Tensor or None
        The tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4) (readonly).
        Return None until spots are initialized.
    centers : torch.Tensor or None
        The tensor of the centers for each roi, of shape (n, 2) (readonly).
        The tensor is of type int, so if the size is odd, the middle is rounded down.
        Return None until spots are initialized.
    file : pathlib.Path or None
        The absolute file path to the image if provided, None otherwise (readonly).
    history : list[str]
        The actions performed on the Diagram from the initialisation (readonly).
    image : torch.Tensor
        The complete brut image of the diagram (readonly).
    rois : torch.Tensor or None
        The tensor of the regions of interest for each spots (readonly).
        For writing, use `self.spots = ...`.
        Return None until spots are initialized. The shape is (n, h, w).
    spots : list[laueimproc.classes.spot.Spot] or None
        All the spots contained in this diagram (read and write).
        Return None until spots are initialized.
    """

    def __init__(
        self,
        data: typing.Union[np.ndarray, torch.Tensor, str, bytes, pathlib.Path],
        *,
        _check: bool = True,
    ):
        """Create a new diagram with appropriated metadata.

        Parameters
        ----------
        data : pathlike or arraylike
            The filename or the array/tensor use as a diagram.
            For memory management, it is better to provide a pathlike rather than an array.
        """
        # declaration
        self._cache: dict[str] = {}  # contains the optional cached data
        self._cache_lock = threading.Lock()  # make the cache acces thread safe
        self._file_or_data: typing.Union[pathlib.Path, torch.Tensor]  # the path to the image file
        self._find_spots_kwargs: typing.Optional[dict] = None  # the kwargs
        self._history: list[str] = []  # the history of the actions performed
        self._rois: typing.Union[None, torch.Tensor, bytes] = None  # rois are undeletable
        self._rois_lock = threading.Lock()  # make the rois compression thread safe
        self._spots: typing.Optional[list[Spot]] = None  # the spots of the diagram

        # initialisation
        if isinstance(data, (str, bytes, pathlib.Path)):
            self._file_or_data = data
            if _check:
                self._file_or_data = pathlib.Path(self._file_or_data).expanduser().resolve()
                assert self._file_or_data.is_file(), self._file_or_data
        else:
            warnings.warn("please provide a path rather than an array", RuntimeWarning)
            self._file_or_data = to_floattensor(data)
            assert self._file_or_data.ndim == 2, self._file_or_data.shape

        # track
        DiagramManager().add_diagram(self)

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
                spot._diagram = None  # to avoid cyclic reference
        with self._rois_lock:
            with self._cache_lock:
                if cache:
                    return (
                        self._file_or_data,
                        self._find_spots_kwargs,
                        self._history,
                        self._rois,
                        spots_no_diagram,
                        self._cache.copy(),
                    )
            return (
                self._file_or_data,
                self._find_spots_kwargs,
                self._history,
                self._rois,
                spots_no_diagram,
            )

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle.

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.
        * Return self by ease.
        """
        (
            self._file_or_data,
            self._find_spots_kwargs,
            self._history,
            self._rois,
            self._spots,
        ) = state[:5]
        if self._spots is not None:
            for spot in self._spots:
                spot._diagram = self
        self._cache = state[5] if len(state) == 6 else {}
        self._cache_lock = threading.Lock()
        self._rois_lock = threading.Lock()
        DiagramManager().add_diagram(self)

    def __str__(self) -> str:
        """Return a nice sumary of the history of this diagram."""

        # title
        if isinstance(self._file_or_data, pathlib.Path):
            text = f"Diagram from {self._file_or_data.name}:"
        else:
            text = f"Diagram from Tensor of id {id(self._file_or_data)}:"

        # history
        if self.spots:
            text += "\n    History:"
            for i, history in enumerate(self._history):
                text += f"\n        {i+1}. {history}"
        else:
            text += "\n    History empty, please initialize the spots `self.find_spots()`."

        # stats
        text += "\n    Current state:"
        text += f"\n        * id, state: {id(self)}, {self.state}"
        if self.is_init():
            text += f"\n        * nbr spots: {len(self.get_spots(_copy=False))}"
        with self._cache_lock, self._rois_lock:
            size = sys.getsizeof(self) + sum(getsizeof(e) for e in self.__dict__.values())
        text += f"\n        * total mem: {bytes2human(size)}"
        return text

    def _find_spots(self, **kwargs):
        """Real version of `find_spots`."""
        # peaks search
        rois, bboxes = peaks_search(self.image, **kwargs)

        # cast into spots objects
        self._rois = rois  # not lock because autoblock
        self._spots = [
            Spot.__new__(Spot).__setstate__((self, index, bbox))
            for index, bbox in enumerate(bboxes.tolist())
        ]
        self._history = [f"{len(self._spots)} spots from self.find_spots(...)"]

    def _set_spots_from_anchors_rois(
        self, anchors: torch.Tensor, rois: list[torch.Tensor], _check: bool = True
    ):
        """Set the new spots from anchors and region of interest.

        Parameters
        ----------
        bboxes : np.ndarray[int]
            The tensor of the bounding boxes of the spots.
        """
        if anchors.shape[0]:
            bboxes = torch.tensor(
                [(i, j, *roi.shape) for (i, j), roi in zip(anchors.tolist(), rois)], dtype=int
            )
        else:
            bboxes = torch.empty((0, 4), dtype=int)
        self._set_spots_from_bboxes(bboxes, _check=_check)
        image = self.image
        with self._rois_lock:
            self._rois = torch.zeros( # zeros 2 times faster than empty + fill 0
                (
                    len(rois),
                    torch.max(bboxes[:, 2]).item() if bboxes.shape[0] else 1,
                    torch.max(bboxes[:, 3]).item() if bboxes.shape[0] else 1,
                ),
                dtype=image.dtype,
                device=image.device,
            )
            for index, (roi, (height, width)) in enumerate(zip(rois, bboxes[:, 2:].tolist())):
                self._rois[index, :height, :width] = roi
        self._history = [f"{len(self._spots)} spots from external anchors and rois"]
        self._find_spots_kwargs = None

    def _set_spots_from_bboxes(self, bboxes: torch.Tensor, _check: bool = True) -> None:
        """Set the new spots from bboxes, in a cleared diagram.

        Parameters
        ----------
        bboxes : np.ndarray[int]
            The tensor of the bounding boxes of the spots.
        """
        if _check:
            image = self.image
            selection = bboxes[:, 0] < 0  # overflow on left
            selection = torch.logical_or(selection, bboxes[:, 1] < 0, out=selection)  # on top
            selection = torch.logical_or(  # on right
                selection, bboxes[:, 0] + bboxes[:, 2] > image.shape[0], out=selection
            )
            selection = torch.logical_or(  # on bottom
                selection, bboxes[:, 0] + bboxes[:, 2] > image.shape[0], out=selection
            )
            if nbr := torch.sum(selection.to(int)).item():
                warnings.warn(f"{nbr} bboxes protrude the image, they are removed", RuntimeWarning)
                bboxes = bboxes[~selection]  # del all overflow bboxes
        self._spots = [
            Spot.__new__(Spot).__setstate__((self, index, (i, j, h, w)))
            for index, (i, j, h, w) in enumerate(bboxes.tolist())
        ]
        with self._rois_lock:
            self._rois = None
        self._history = [f"{len(self._spots)} spots from external bboxes"]
        self._find_spots_kwargs = None

    def _set_spots_from_spots(self, new_spots: list[Spot], _check: bool = True) -> None:
        """Set the new spots, in a cleared diagram (not cleard for self._rois).

        Parameters
        ----------
        new_spots : list[Spot]
            All the new spots to set, they can to comes from any diagram.
        """
        # verification
        image = self.image
        if _check:  # very slow
            new_spots_ = [
                s for s in new_spots
                if (
                    s.bbox[0] >= 0  # overflow on left
                    and s.bbox[1] >= 0   # on top
                    and s.bbox[0]+s.bbox[2] <= image.shape[0]  # on right
                    and s.bbox[1]+s.bbox[3] <= image.shape[1]  # on bottom
                )
            ]
            if nbr := len(new_spots) - len(new_spots_):
                warnings.warn(f"{nbr} spots protrude the image, they are removed", RuntimeWarning)
                new_spots = new_spots_

        # set new spots
        rois = [  # extract rois before change index and reset self._roi in case of ref to self
            s.roi for s in new_spots
        ]
        for index, spot in enumerate(new_spots):
            spot._diagram, spot._index = self, index
        self._spots = new_spots
        bboxes = self.bboxes  # reachable because self._spots is defined
        with self._rois_lock:
            self._rois = torch.zeros(  # zeros 2 times faster than empty + fill 0
                (
                    bboxes.shape[0],
                    torch.max(bboxes[:, 2]).item() if bboxes.shape[0] else 1,
                    torch.max(bboxes[:, 3]).item() if bboxes.shape[0] else 1,
                ),
                dtype=image.dtype,
                device=image.device,
            )
            for index, (roi, (height, width)) in enumerate(zip(rois, bboxes[:, 2:].tolist())):
                self._rois[index, :height, :width] = roi
        self._history = [f"{len(self._spots)} spots from external spots"]
        self._find_spots_kwargs = None

    @property
    def bboxes(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width)."""
        @auto_cache
        def _compute_bboxes(self) -> torch.Tensor:
            """Helper for `self.bboxes`."""
            if self.spots:
                return torch.tensor([s.bbox for s in self.spots], dtype=int)
            return torch.empty((0, 4), dtype=int)
        if not self.is_init():
            return None
        return _compute_bboxes(self)

    @property
    def centers(self) -> typing.Union[None, torch.Tensor]:  # very fast -> no cache
        """Return the tensor of the centers for each roi."""
        if not self.is_init():
            return None
        if self.spots:
            bboxes = self.bboxes
            return bboxes[:, :2] + bboxes[:, 2:]//2
        return torch.empty((0, 2), dtype=int)

    def clone(self, deep: bool = True, cache: bool = True):
        """Instanciate a new identical diagram.

        Parameters
        ----------
        deep : boolean, default=True
            If True, the memory of the new created diagram object is totally
            independ of this object (slow but safe). Otherwise, `image` and `rois` attributes
            and cached big data share the same memory view (Tensor). So modifying one of these
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

    def compress(self, size: numbers.Real = math.inf, *, _levels: set[int] = None) -> int:
        """Delete or compress attributes and elements in the cache.

        Paremeters
        ----------
        size : int
            The quantity of bytes to remove from the cache.

        Returns
        -------
        removed : int
            The number of bytes removed from the cache.
        """
        # verifiactions
        assert isinstance(size, numbers.Real), size.__class__.__name__
        assert size > 0, size

        _levels = _levels or {0, 1, 2}

        # declaration
        removed = 0

        # delete obsolete cache
        if 0 in _levels:
            pattern = r"(?P<state>[0-9a-f]{32})\.\w+\([0-9a-f]{32}\)"
            state = self.state
            with self._cache_lock:
                for key in list(self._cache):  # copy keys for pop
                    if (match := re.search(pattern, key)) is None:
                        continue
                    if match["state"] != state:
                        removed += sys.getsizeof(key) + getsizeof(self._cache.pop(key))

        # delete valid cache
        if 1 in _levels:
            with self._cache_lock:
                size_to_key = {getsizeof(v): k for k, v in self._cache.items()}
                for item_size in sorted(size_to_key, reverse=True):  # delete biggest elements first
                    key = size_to_key[item_size]
                    removed += sys.getsizeof(key) + getsizeof(self._cache.pop(key))
                    if removed >= size:
                        return removed

        # compress undeletable cache
        if 2 in _levels:
            with self._rois_lock:
                if isinstance(self._rois, torch.Tensor):
                    removed += getsizeof(self._rois)
                    self._rois = compress_rois(self._rois)
                    removed -= sys.getsizeof(self._rois)
                    if removed >= size:
                        return removed

        return removed

    @property
    def file(self) -> typing.Union[None, pathlib.Path]:
        """Return the absolute file path to the image, if it is provided."""
        if isinstance(self._file_or_data, pathlib.Path):
            return self._file_or_data
        return None

    def filter_spots(
        self, indexs: typing.Container, msg: str = "general filter", *, inplace: bool = False
    ):
        """Keep only the given spots, delete the rest.

        This method can be used for filtering or sorting spots.

        Parameters
        ----------
        indexs : arraylike
            The list of the indexs of the spots to keep
            or the boolean vector with True for keeping the spot, False otherwise like a mask.
        msg : str
            The message to happend to the history.
        inplace : boolean, default=True
            If True, modify the diagram self (no copy) and return a reference to self.
            If False, first clone the diagram, then apply the selection on the new diagram,
            It create a checkpoint (real copy) (but it is slowler).

        Returns
        -------
        filtered_diagram : BaseDiagram
            Return self if inplace is True or a modified clone of self otherwise.
        """
        # verifications and cast
        if not self.is_init():
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`) before to filter it"
            )
        assert hasattr(indexs, "__iter__"), indexs.__class__.__name__
        assert isinstance(msg, str), msg.__class__.__name__
        assert isinstance(inplace, bool), inplace.__class__.__name__
        indexs = torch.as_tensor(indexs)
        indexs = torch.squeeze(indexs)
        assert indexs.ndim == 1, f"only a 1d vector is accepted, shape is {indexs.shape}"
        if indexs.dtype is torch.bool:  # case mask -> convert into index list
            assert indexs.shape[0] == len(self.spots), (
                "the mask has to have the same length as the number of spots, "
                f"there are {len(self.spots)} spots and mask is of len {indexs.shape[0]}"
            )
            indexs = torch.arange(len(self.spots))[indexs]  # bool -> indexs
        else:
            # assert len(set(indexs.tolist())) == indexs.shape[0], \  # very slow !
            #     "each index must be unique, but some of them appear more than once"
            assert not indexs.shape[0] or torch.min(indexs).item() >= 0, \
                "negative index is not allowed"
            assert not indexs.shape[0] or torch.max(indexs).item() < len(self.spots), \
                "some indexes are out of range"

        # manage inplace
        nb_spots = len(self.spots)
        if not inplace:
            self = self.clone()  # pylint: disable=W0642

        # update history, it has to be done before changing state to be catched by signature
        self._history.append(f"{nb_spots} to {len(indexs)} spots: {msg}")

        # filter the spots
        self._spots = [self._spots[new_index] for new_index in indexs.tolist()]
        for new_index, spot in enumerate(self._spots):
            spot._index = new_index  # pylint: disable=W0212

        # filter the rois
        rois = self.rois[indexs]
        if len(indexs):
            bboxes = self.bboxes
            height = torch.max(bboxes[:, 2]).item()
            width = torch.max(bboxes[:, 3]).item()
            rois = rois[:, :height, :width]
        with self._rois_lock:
            self._rois = rois

        return None if inplace else self

    def find_spots(self, **kwargs) -> None:
        """Search all the spots in this diagram, store the result into the `spots` attribute.

        Parameters
        ----------
        kwargs : dict
            Transmitted to ``laueimproc.improc.peaks_search.peaks_search``.
        """
        self._find_spots_kwargs = kwargs
        self._history = ["x spots from self.find_spots(...)"]
        with self._rois_lock:
            self._rois = None
        self._spots = None

    def get_spots(self, *, _copy: bool = True) -> list[Spot]:
        """Return the spots in a unordered set container."""
        if self._spots is None and self._find_spots_kwargs is not None:
            self._find_spots(**self._find_spots_kwargs)
        if self._spots is None:  # case spots are never been searched
            return None
        # copy is slow but it is a strong protection against the user
        return self._spots.copy() if _copy else self._spots

    @property
    def history(self) -> list[str]:
        """Return the actions performed on the Diagram from the initialisation."""
        if not self.is_init():
            return []
        self.get_spots(_copy=False)  # update hystory (first step init)
        return self._history.copy()  # copy for user protection

    def is_init(self) -> bool:
        """Return True if the diagram has been initialized."""
        return self._spots is not None or self._find_spots_kwargs is not None

    @property
    def image(self) -> torch.Tensor:
        """Return the complete image of the diagram."""
        with self._cache_lock:
            if "image" not in self._cache:  # no auto ache because it is state invariant
                self._cache["image"] = (
                    read_image(self._file_or_data)
                    if isinstance(self._file_or_data, pathlib.Path) else
                    self._file_or_data
                )
            return self._cache["image"]

    def plot(
        self,
        disp: typing.Optional[typing.Union[Figure, Axes]] = None,
        vmin: typing.Optional[numbers.Real] = None,
        vmax: typing.Optional[numbers.Real] = None,
    ) -> Axes:
        """Prepare for display the diagram and the spots.

        Parameters
        ----------
        disp : matplotlib.figure.Figure or matplotlib.axes.Axes
            The matplotlib figure to complete.
        vmin : float, optional
            The minimum intensity ploted.
        vmax : float, optional
            The maximum intensity ploted.

        Notes
        -----
        It doesn't create the figure and call show.
        Use `self.show()` to Display the diagram from scratch.
        """
        assert disp is None or isinstance(disp, (Figure, Axes))
        image = self.image
        if vmin is None:
            vmin = torch.min(image).item()
        assert vmin is None or isinstance(vmin, numbers.Real), vmin.__class__.__name__
        if vmax is None:
            vmax = torch.mean(image).item() + 5.0*torch.std(image).item()
        assert isinstance(vmax, numbers.Real), vmax.__class__.__name__

        # fill figure metadata
        axes = disp  # is gonna changed
        disp = disp or Figure(layout="tight")
        if isinstance(disp, Figure):
            if isinstance(self._file_or_data, pathlib.Path):
                disp.suptitle(f"Diagram {self._file_or_data.name}")
            else:
                disp.suptitle(f"Diagram from Tensor of id {id(self._file_or_data)}")
            axes = disp.add_subplot()

        # fill axes
        axes.set_ylabel("i (first axis)")
        axes.set_xlabel("j (second axis)")
        axes.imshow(
            image.numpy(force=True).transpose(),
            aspect="equal",
            extent=(0, self.image.shape[1], self.image.shape[0], 0),  # origin to corner of pxl
            interpolation=None,  # antialiasing is True
            norm="log",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        if self.spots:
            bboxes = self.bboxes.numpy(force=True)
            axes.plot(
                np.vstack((
                    bboxes[:, 0],
                    bboxes[:, 0]+bboxes[:, 2],
                    bboxes[:, 0]+bboxes[:, 2],
                    bboxes[:, 0],
                    bboxes[:, 0],
                )),
                np.vstack((
                    bboxes[:, 1],
                    bboxes[:, 1],
                    bboxes[:, 1]+bboxes[:, 3],
                    bboxes[:, 1]+bboxes[:, 3],
                    bboxes[:, 1],
                )),
                color="blue",
                scalex=False,
                scaley=False,
            )
        return axes

    @property
    def rois(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the rois of the spots."""
        if not self.is_init():
            return None
        with self._rois_lock:
            if self._rois is None:
                image = self.image
                bboxes = self.bboxes
                self._rois = torch.zeros(  # zeros 2 times faster than empty + fill 0
                    (
                        bboxes.shape[0],
                        torch.max(bboxes[:, 2]).item() if bboxes.shape[0] else 1,
                        torch.max(bboxes[:, 3]).item() if bboxes.shape[0] else 1,
                    ),
                    dtype=image.dtype,
                    device=image.device,
                )
                for index, (i, j, height, width) in enumerate(bboxes.tolist()):
                    self._rois[index, :height, :width] = image[i:i+height, j:j+width]
            if isinstance(self._rois, bytes):
                self._rois = decompress_rois(self._rois)  # take 6 ms in average
                # return decompress_rois(self._rois)  # better no store decompress for sparse memory
            return self._rois

    def set_spots(self, new_spots: typing.Container) -> None:
        """Set the new spots as the current spots, reset the history and the cache.

        Paremeters
        ----------
        new_spots : typing.Container
            * Can be an iterable (list, tuple, set, ...) of `Spot` instances.
            * Can be an arraylike of bounding boxes n * (anchor_i, anchor_j, height, width)
            * Can be a tuple of anchors (arraylike of shape n * (anchor_i, anchor_i))
                and rois (iterable of images patches).
            * Can be None for reset all (same as calling the `reset` method)
        """
        assert hasattr(new_spots, "__iter__"), new_spots.__class__.__name__

        # clear history and internal state (self._rois is cleared later)
        with self._cache_lock:
            self._cache = {}
        self._history = []
        self._spots = None

        # case tensor
        if isinstance(new_spots, np.ndarray):
            new_spots = new_spots.from_numpy(new_spots)
        if isinstance(new_spots, torch.Tensor):
            if new_spots.dtype.is_floating_point:
                new_spots = new_spots.to(int)
            if new_spots.ndim == 2 and new_spots.shape[1] == 4:  # case from bounding boxes
                return self._set_spots_from_bboxes(new_spots)

            raise NotImplementedError(
                f"impossible to set new spots from an array of shape {new_spots.shape}, "
                "if has to be of shape (n, 4) for bounding boxes"
            )

        # preparation of data
        if not isinstance(new_spots, list):
            new_spots = list(new_spots)  # freeze the order

        # case from Spot instances
        cls = {s.__class__ for s in new_spots}
        if not cls or (len(cls) == 1 and next(iter(cls)) is Spot):  # catch empty case
            return self._set_spots_from_spots(new_spots)  # reset self._rois

        # case from anchors, rois
        if (
            len(new_spots) == 2
            and isinstance(new_spots[0], (list, tuple, np.ndarray, torch.Tensor))
            and isinstance(new_spots[1], (list, tuple))
            and len(new_spots[0]) == len(new_spots[1])
        ):
            if not isinstance(new_spots[0], torch.Tensor):  # no torch.as_array memory leak
                new_spots[0] = torch.as_tensor(new_spots[0], dtype=int)
            anchors = new_spots[0]
            rois = [to_floattensor(roi) for roi in new_spots[1]]
            return self._set_spots_from_anchors_rois(anchors, rois)

        # case tensor (recursive delegation)
        if len(cls) == 1 and issubclass(next(iter(cls)), (list, tuple)):
            return self.set_spots(torch.tensor(new_spots, dtype=int))

        raise NotImplementedError(
            f"impossible to set new spots from {new_spots}, "
            "it has to be a container of Spot instance "
            "or a container of bounding boxes"
        )

    @property
    def spots(self) -> typing.Union[list[Spot]]:
        """Alias to the `get_spots`."""
        return self.get_spots()

    @spots.setter
    def spots(self, new_spots: typing.Container):
        """Alias to ``set_spots``."""
        self.set_spots(new_spots)

    @property
    def state(self) -> str:
        """Return a hash of the diagram.

        If two diagrams gots the same state, it means they are the same.
        The hash take in consideration the internal state of the diagram.
        The retruned value is a hexadecimal strinf of length 32
        """
        hasher = hashlib.md5(usedforsecurity=False)
        if isinstance(self._file_or_data, pathlib.Path):
            hasher.update(str(self._file_or_data.name).encode())
        else:
            hasher.update(id(self._file_or_data).to_bytes(8, "big"))
        hasher.update(str(self._find_spots_kwargs).encode())
        hasher.update("\n".join(self._history[1:]).encode())
        return hasher.hexdigest()
