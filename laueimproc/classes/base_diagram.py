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
from laueimproc.opti.cache import getsizeof
from laueimproc.opti.manager import DiagramManager
from laueimproc.opti.rois import filter_by_indexs
from laueimproc.opti.rois import imgbboxes2raw
from laueimproc.opti.rois import rawshapes2rois
from .spot import Spot


class BaseDiagram:
    """A Basic diagram with the fondamental structure.

    Attributes
    ----------
    bboxes : torch.Tensor or None
        The int32 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
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
    rawrois : torch.Tensor or None
        The tensor of the raw regions of interest for each spots (readonly).
        Return None until spots are initialized. The shape is (n, h, w).
        Contrary to `self.rois`, it is only a view of `self.image`.
    rois : torch.Tensor or None
        The tensor of the provided regions of interest for each spots (readonly).
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
        self._rois: typing.Optional[tuple[bytearray, torch.Tensor]] = None  # datarois, bboxes
        self._rois_lock = threading.Lock()  # make the rois acces thread safe

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
        with self._rois_lock:
            rois = None if self._rois is None else (self._rois[0].copy(), self._rois[1].clone())
        if cache:
            with self._cache_lock:
                return (
                    self._file_or_data,
                    self._find_spots_kwargs,
                    self._history,
                    rois,
                    self._cache.copy(),
                )
        return (
            self._file_or_data,
            self._find_spots_kwargs,
            self._history,
            rois,
        )

    def __len__(self) -> int:
        """Return the nbr of spots or 0."""
        self.flush()
        try:  # it doesn't matter if acces is not thread safe
            return len(self._rois[1])
        except TypeError:
            return 0

    def __repr__(self) -> str:
        """Give a very compact excecutable representation."""
        if self.file is None:
            return f"{self.__class__.__name__}(Tensor(...))"
        return f"{self.__class__.__name__}({self.file.name})"

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle.

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.
        * Return self by ease.
        """
        # not verification for thread safe
        # because this method is never meant to be call fom a thread.
        (
            self._file_or_data,
            self._find_spots_kwargs,
            self._history,
            self._rois,
        ) = state[:4]
        self._cache = state[4] if len(state) == 5 else {}
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
        if self.is_init():
            text += "\n    History:"
            for i, history in enumerate(self.history):  # not ._history to be updated
                text += f"\n        {i+1}. {history}"
        else:
            text += "\n    History empty, please initialize the spots `self.find_spots()`."

        # stats
        text += "\n    Current state:"
        text += f"\n        * id, state: {id(self)}, {self.state}"
        if self.is_init():
            text += f"\n        * nbr spots: {len(self)}"
        with self._cache_lock, self._rois_lock:
            size = sys.getsizeof(self) + sum(getsizeof(e) for e in self.__dict__.values())
        text += f"\n        * total mem: {bytes2human(size)}"
        return text

    def _find_spots(self, **kwargs):
        """Real version of `find_spots`."""
        datarois, bboxes = peaks_search(self.image, **kwargs)
        with self._rois_lock:
            self._rois = (datarois, bboxes)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        self._history = [f"{len(self)} spots from self.find_spots({kwargs_str})"]

    def _set_spots_from_anchors_rois(self, anchors: torch.Tensor, rois: list[torch.Tensor]):
        """Set the new spots from anchors and regions of interest."""
        if anchors.shape[0]:
            bboxes = torch.tensor(
                [(i, j, *roi.shape) for (i, j), roi in zip(anchors.tolist(), rois)],
                dtype=torch.int32,
            )
            flat_rois = np.concatenate(
                [
                    roi.numpy(force=True).ravel()
                    for roi, (h, w) in zip(rois, bboxes[:, 2:].tolist())
                ],
                dtype=np.float32,
            )
            datarois = bytearray(flat_rois.tobytes())
        else:
            bboxes = torch.empty((0, 4), dtype=torch.int32)
            datarois = bytearray(b"")
        with self._rois_lock:
            self._rois = (datarois, bboxes)
        self._history = [f"{len(self)} spots from external anchors and rois"]
        self._find_spots_kwargs = None

    def _set_spots_from_bboxes(self, bboxes: torch.Tensor):
        """Set the new spots from bboxes, in a cleared diagram."""
        datarois = imgbboxes2raw(self.image, bboxes)
        with self._rois_lock:
            self._rois = (datarois, bboxes)
        self._history = [f"{len(self)} spots from external bboxes"]
        self._find_spots_kwargs = None

    def _set_spots_from_spots(self, new_spots: list[Spot]):
        """Set the new spots, from Spots instances."""
        rois = [s.roi for s in new_spots]
        anchors = torch.tensor([s.anchor for s in new_spots], dtype=torch.int32)
        self._set_spots_from_anchors_rois(anchors, rois)
        self._history[-1] = f"{len(self)} spots from external spots"

    @property
    def bboxes(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width)."""
        if not self.is_init():
            return None
        self.flush()
        with self._rois_lock:
            bboxes = self._rois[1]
        return bboxes.clone()

    @property
    def centers(self) -> typing.Union[None, torch.Tensor]:  # very fast -> no cache
        """Return the tensor of the centers for each roi."""
        if not self.is_init():
            return None
        if len(self):
            bboxes = self.bboxes
            return bboxes[:, :2] + bboxes[:, 2:]//2
        return torch.empty((0, 2), dtype=torch.int32)

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
        """Delete attributes and elements in the cache.

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

        _levels = _levels or {0, 1}

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
            The list of the indexs of the spots to keep (negatives indexs are allow),
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
            Return Nonw if inplace is True, or a filtered clone of self otherwise.
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
            assert indexs.shape[0] == len(self), (
                "the mask has to have the same length as the number of spots, "
                f"there are {len(self)} spots and mask is of len {indexs.shape[0]}"
            )
            indexs = torch.arange(len(self), dtype=torch.int64)[indexs]  # bool -> indexs
        elif indexs.dtype != torch.int64:
            indexs = indexs.to(torch.int64)

        # manage inplace
        nb_spots = len(self)  # flush in background
        if not inplace:
            self = self.clone()  # pylint: disable=W0642

        # update history, it has to be done before changing state to be catched by signature
        self._history.append(f"{nb_spots} to {len(indexs)} spots: {msg}")

        # apply filter
        with self._rois_lock:
            self._rois = filter_by_indexs(indexs, *self._rois)

        return None if inplace else self

    def find_spots(self, **kwargs) -> None:
        """Search all the spots in this diagram, store the result into the `spots` attribute.

        Parameters
        ----------
        kwargs : dict
            Transmitted to ``laueimproc.improc.peaks_search.peaks_search``.
        """
        self._find_spots_kwargs = kwargs
        self._history = ["x spots from self.find_spots(...)"]  # gonna be updated by _find_spots
        with self._rois_lock:
            self._rois = None

    def flush(self):
        """Perform all pending calculations."""
        if self._rois is None and self._find_spots_kwargs is not None:
            self._find_spots(**self._find_spots_kwargs)

    @property
    def history(self) -> list[str]:
        """Return the actions performed on the Diagram since the initialisation."""
        if not self.is_init():
            return []
        self.flush()
        return self._history.copy()  # copy for user protection

    def is_init(self) -> bool:
        """Return True if the diagram has been initialized."""
        return self._rois is not None or self._find_spots_kwargs is not None

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
        **kwargs,
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
        show_axis: boolean, default=True
            Display the label and the axis if True.
        show_bboxes: boolean, default=True
            Draw all the bounding boxes if True.
        show_image: boolean, default=True
            Display the image if True, dont call imshow otherwise.

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
        assert isinstance(kwargs.get("show_axis", True), bool), kwargs["show_axis"]
        assert isinstance(kwargs.get("show_bboxes", True), bool), kwargs["show_bboxes"]
        assert isinstance(kwargs.get("show_image", True), bool), kwargs["show_image"]

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
        if kwargs.get("show_axis", True):
            axes.set_xlabel("x (first dim in 'xy' conv)")
            axes.xaxis.set_label_position("bottom")
            axes.xaxis.set_ticks_position("bottom")
            axes.set_ylabel("y (second dim in 'xy' conv)")
            axes.yaxis.set_label_position("right")
            axes.yaxis.set_ticks_position("right")
            axes.secondary_xaxis(
                "top", functions=(lambda x: x-.5, lambda j: j+.5)
            ).set_xlabel("j (second dim in 'ij' conv)")
            axes.secondary_yaxis(
                "left", functions=(lambda y: y-.5, lambda i: i+.5)
            ).set_ylabel("i (first dim in 'ij' conv)")
        if kwargs.get("show_image", True):
            axes.imshow(
                image.numpy(force=True).transpose(),
                aspect="equal",
                extent=(0.5, self.image.shape[0]+.5, self.image.shape[1]+.5, .5),
                interpolation=None,  # antialiasing is True
                norm="log",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
        if kwargs.get("show_boxes", True) and len(self):
            bboxes = self.bboxes.numpy(force=True).astype(np.float32)
            bboxes[:, :2] += 0.5
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
    def rawrois(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the raw rois of the spots."""
        if not self.is_init():
            return None
        with self._rois_lock:
            _, bboxes = self._rois
        return rawshapes2rois(imgbboxes2raw(self.image, bboxes), bboxes[:, 2:].numpy(force=True))

    @property
    def rois(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the provided rois of the spots."""
        if not self.is_init():
            return None
        self.flush()
        with self._rois_lock:
            datarois, bboxes = self._rois
        return rawshapes2rois(datarois, bboxes[:, 2:].numpy(force=True))

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

        # case tensor
        if isinstance(new_spots, np.ndarray):
            new_spots = new_spots.from_numpy(new_spots)
        if isinstance(new_spots, torch.Tensor):
            if new_spots.dtype.is_floating_point:
                new_spots = new_spots.to(torch.int32)
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
                new_spots[0] = torch.as_tensor(new_spots[0], dtype=torch.int32)
            anchors = new_spots[0]
            rois = [to_floattensor(roi) for roi in new_spots[1]]
            return self._set_spots_from_anchors_rois(anchors, rois)

        # case tensor (recursive delegation)
        if len(cls) == 1 and issubclass(next(iter(cls)), (list, tuple)):
            return self.set_spots(torch.tensor(new_spots, dtype=torch.int32))

        raise NotImplementedError(
            f"impossible to set new spots from {new_spots}, "
            "it has to be a container of Spot instance "
            "or a container of bounding boxes"
        )

    @property
    def spots(self) -> typing.Union[None, list[Spot]]:
        """Return the list of the spots."""
        if (bboxes := self.bboxes) is None:
            return None
        return [Spot.__new__(Spot).__setstate__((self, i)) for i in range(len(self))]

    @spots.setter
    def spots(self, new_spots: typing.Container):
        """Alias to ``set_spots``."""
        self.set_spots(new_spots)

    @property
    def state(self) -> str:
        """Return a hash of the diagram.

        If two diagrams gots the same state, it means they are the same.
        The hash take in consideration the internal state of the diagram.
        The retruned value is a hexadecimal string of length 32.
        """
        hasher = hashlib.md5(usedforsecurity=False)
        if isinstance(self._file_or_data, pathlib.Path):
            hasher.update(str(self._file_or_data.name).encode())
        else:
            hasher.update(id(self._file_or_data).to_bytes(8, "big"))
        hasher.update(str(self._find_spots_kwargs).encode())  # order same since python3.7
        hasher.update("\n".join(self._history[1:]).encode())
        return hasher.hexdigest()
