#!/usr/bin/env python3

"""Define the pytonic structure of a basic Diagram."""

import functools
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

import numpy as np
import torch

from laueimproc.common import bytes2human
from laueimproc.improc.peaks_search import peaks_search
from laueimproc.io.read import read_image, to_floattensor
from laueimproc.opti.cache import getsizeof
from laueimproc.opti.manager import DiagramManager
from laueimproc.opti.rois import filter_by_indices
from laueimproc.opti.rois import imgbboxes2raw
from laueimproc.opti.rois import rawshapes2rois


def check_init(meth: typing.Callable) -> typing.Callable:
    """Decorate a Diagram method to ensure the diagram has been init."""
    assert callable(meth), meth.__class__.__name__

    @functools.wraps(meth)
    def check_init_meth(diagram, *args, **kwargs):
        if not diagram.is_init():
            raise RuntimeError(
                f"before calling the {meth} diagram method, initialize the diagram "
                f"{repr(diagram)} by invoking 'find_spots' for example"
            )
        return meth(diagram, *args, **kwargs)

    return check_init_meth


class BaseDiagram:
    """A Basic diagram with the fondamental structure.

    Attributes
    ----------
    bboxes : torch.Tensor or None
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4) (readonly).
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
    """

    def __init__(
        self,
        data: typing.Union[np.ndarray, torch.Tensor, str, pathlib.Path],
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
        self._cache: tuple[threading.Lock, dict[str]] = (  # contains the optional cached data
            threading.Lock(), {}
        )
        self._file_or_data: typing.Union[pathlib.Path, torch.Tensor]  # the path to the image file
        self._find_spots_kwargs: typing.Optional[dict] = None  # the kwargs
        self._history: list[str] = []  # the history of the actions performed
        self._properties: dict[str, tuple[typing.Union[None, str], object]] = {}  # the properties
        self._rois: typing.Optional[tuple[bytearray, torch.Tensor]] = None  # datarois, bboxes
        self._rois_lock = threading.Lock()  # make the rois acces thread safe

        # initialisation
        if isinstance(data, (str, pathlib.Path)):
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
        """Make the diagram pickleable."""
        with self._rois_lock:
            rois = None if self._rois is None else (self._rois[0].copy(), self._rois[1].clone())
        if cache:
            with self._cache[0]:
                return (
                    self._file_or_data,
                    self._find_spots_kwargs,
                    self._history,
                    self._properties,
                    rois,
                    self._cache[1].copy(),
                )
        return (
            self._file_or_data,
            self._find_spots_kwargs,
            self._history,
            self._properties,
            rois,
        )

    def __len__(self) -> int:
        """Return the nbr of spots or 0.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> len(diagram)
        0
        >>> diagram.find_spots()
        >>> len(diagram)
        219
        >>>
        """
        self.flush()
        try:  # it doesn't matter if acces is not thread safe
            return len(self._rois[1])
        except TypeError:
            return 0

    def __repr__(self) -> str:
        """Give a very compact representation.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> BaseDiagram(get_sample())
        BaseDiagram(ge.jp2)
        >>>
        """
        if self.file is None:
            return f"{self.__class__.__name__}(Tensor(...))"
        return f"{self.__class__.__name__}({self.file.name})"

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle.

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.

        Examples
        --------
        >>> import pickle
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram_bis = pickle.loads(pickle.dumps(diagram))
        >>> assert id(diagram) != id(diagram_bis)
        >>> assert diagram.state == diagram_bis.state
        >>>
        """
        # not verification for thread safe
        # because this method is never meant to be call from a thread.
        (
            self._file_or_data,
            self._find_spots_kwargs,
            self._history,
            self._properties,
            self._rois,
        ) = state[:5]
        self._cache = (threading.Lock(), (state[5] if len(state) == 6 else {}))
        self._rois_lock = threading.Lock()
        DiagramManager().add_diagram(self)

    def __str__(self) -> str:
        """Return a nice sumary of the history of this diagram.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> print(BaseDiagram(get_sample()))  # doctest: +ELLIPSIS
        Diagram from ge.jp2:
            History empty, please initialize the spots `self.find_spots()`.
            No Properties
            Current state:
                * id, state: ...
                * total mem: 536.0B
        >>>
        """
        # title
        text = (
            f"Diagram from {self._file_or_data.name}:"
            if isinstance(self._file_or_data, pathlib.Path) else
            f"Diagram from Tensor of id {id(self._file_or_data)}:"
        )

        # history
        if self.is_init():
            text += "\n    History:"
            for i, history in enumerate(self.history):  # not ._history to be updated
                text += f"\n        {i+1}. {history}"
        else:
            text += "\n    History empty, please initialize the spots `self.find_spots()`."

        # properties
        properties = []
        for name in sorted(self._properties):
            try:
                properties.append((name, self.get_property(name)))
            except KeyError:
                pass
        if properties:
            text += "\n    Properties:"
            for name, value in properties:
                if len(value_str := str(value).replace("\n", "\\n")) >= 80:
                    value_str = f"<{value.__class__.__name__} object>"
                text += f"\n        * {name}: {value_str}"
        else:
            text += "\n    No Properties"

        # stats
        text += "\n    Current state:"
        text += f"\n        * id, state: {id(self)}, {self.state}"
        if self.is_init():
            text += f"\n        * nbr spots: {len(self)}"
        with self._cache[0], self._rois_lock:
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
        """Set the new spots from anchors and regions of interest.

        Examples
        --------
        >>> import itertools, random
        >>> import numpy as np
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> anchors = list(  # numpy convention
        ...     itertools.product(range(15, min(diagram.image.shape)-30, 200), repeat=2)
        ... )
        >>> rois = [  # roi patches, as little images
        ...     np.empty((random.randint(5, 30), random.randint(5, 30)), np.uint16)
        ...     for _ in anchors
        ... ]
        >>> diagram.set_spots((anchors, rois))
        >>> len(diagram)
        100
        >>> diagram.rois.min() >= 0 and diagram.rois.max() <= 1  # range in [0, 1]
        tensor(True)
        >>>
        """
        if anchors.shape[0]:
            bboxes = torch.tensor(
                [(i, j, *roi.shape) for (i, j), roi in zip(anchors.tolist(), rois)],
                dtype=torch.int16,
            )
            flat_rois = np.concatenate(
                [
                    roi.numpy(force=True).ravel()
                    for roi, (h, w) in zip(rois, bboxes[:, 2:].tolist())
                ],
            ).astype(np.float32, copy=False)  # not 100% shure float32
            datarois = bytearray(flat_rois.tobytes())
        else:
            bboxes = torch.empty((0, 4), dtype=torch.int16)
            datarois = bytearray(b"")
        with self._rois_lock:
            self._rois = (datarois, bboxes)
        self._history = [f"{len(self)} spots from external anchors and rois"]
        self._find_spots_kwargs = None

    def _set_spots_from_bboxes(self, bboxes: torch.Tensor):
        """Set the new spots from bboxes, in a cleared diagram.

        Examples
        --------
        >>> import itertools, random
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> bboxes = [  # numpy convention (*anchor, *shape)
        ...     (i, j, random.randint(5, 30), random.randint(5, 30))
        ...     for i, j in itertools.product(range(15, min(diagram.image.shape)-30, 200), repeat=2)
        ... ]
        >>> diagram.set_spots(bboxes)
        >>> len(diagram)
        100
        >>> diagram.rois.min() >= 0 and diagram.rois.max() <= 1  # range in [0, 1]
        tensor(True)
        >>>
        """
        datarois = imgbboxes2raw(self.image, bboxes)
        with self._rois_lock:
            self._rois = (datarois, bboxes)
        self._history = [f"{len(self)} spots from external bboxes"]
        self._find_spots_kwargs = None

    def add_property(self, name: str, value: object, *, erasable: bool = True):
        """Add a property to the diagram.

        Parameters
        ----------
        name : str
            The identifiant of the property for the requests.
            If the property is already defined with the same name, the new one erase the older one.
        value
            The property value. If a number is provided, it will be faster.
        erasable : boolean, default=True
            If set to False, the property will be set in stone,
            overwise, the property will desappear as soon as the diagram state changed.
        """
        assert isinstance(name, str), name.__class__.__name__
        assert isinstance(erasable, bool), erasable.__class__.__name__
        with self._cache[0]:
            self._properties[name] = ((self.state if erasable else None), value)

    @property
    def bboxes(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width).

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> print(diagram.bboxes)
        None
        >>> diagram.find_spots()
        >>> print(diagram.bboxes)  # doctest: +ELLIPSIS
        tensor([[   0,    0,   69,   17],
                [   0,   20,    3,   12],
                [   0, 1948,    3,   15],
                ...,
                [1904,  180,   16,   15],
                [1931, 1968,   13,   13],
                [1964, 1170,   23,   19]], dtype=torch.int16)
        >>>
        """
        if not self.is_init():
            return None
        self.flush()
        with self._rois_lock:
            bboxes = self._rois[1]
        return bboxes.clone()

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

        Returns
        -------
        Diagram
            The new copy of self.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram_bis = diagram.clone()
        >>> assert id(diagram) != id(diagram_bis)
        >>> assert diagram.state == diagram_bis.state
        >>>
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
            with self._cache[0]:
                for key in list(self._cache[1]):  # copy keys for pop
                    if (match := re.search(pattern, key)) is None:
                        continue
                    if match["state"] != state:
                        removed += sys.getsizeof(key) + getsizeof(self._cache[1].pop(key))
                # for key in list(self._properties):
                #     match = self._properties[key][0]
                #     if match is not None and match != state:
                #         removed += sys.getsizeof(key) + getsizeof(self._properties.pop(key))

        # delete valid cache
        if 1 in _levels:
            with self._cache[0]:
                size_to_key = {getsizeof(v): k for k, v in self._cache[1].items()}
                for item_size in sorted(size_to_key, reverse=True):  # delete biggest elements first
                    key = size_to_key[item_size]
                    removed += sys.getsizeof(key) + getsizeof(self._cache[1].pop(key))
                    if removed >= size:
                        return removed

        return removed

    @functools.cached_property
    def file(self) -> typing.Union[None, pathlib.Path]:
        """Return the absolute file path to the image, if it is provided.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> BaseDiagram(get_sample()).file  # doctest: +ELLIPSIS
        PosixPath('/.../laueimproc/io/ge.jp2')
        >>>
        """
        return self._file_or_data if isinstance(self._file_or_data, pathlib.Path) else None

    @check_init
    def filter_spots(
        self, criteria: typing.Container, msg: str = "general filter", *, inplace: bool = True
    ):
        """Keep only the given spots, delete the rest.

        This method can be used for filtering or sorting spots.

        Parameters
        ----------
        criteria : arraylike
            The list of the indices of the spots to keep (negatives indices are allow),
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
            Return None if inplace is True, or a filtered clone of self otherwise.

        Examples
        --------
        >>> from pprint import pprint
        >>> import torch
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.find_spots()
        >>> indices = torch.arange(0, len(diagram), 2)
        >>> diagram.filter_spots(indices, "keep even spots")
        >>> cond = diagram.bboxes[:, 1] < diagram.image.shape[1]//2
        >>> diag_final = diagram.filter_spots(cond, "keep spots on left", inplace=False)
        >>> pprint(diagram.history)
        ['219 spots from self.find_spots()', '219 to 110 spots: keep even spots']
        >>> pprint(diag_final.history)
        ['219 spots from self.find_spots()',
         '219 to 110 spots: keep even spots',
         '110 to 62 spots: keep spots on left']
        >>>
        """
        # verifications and cast
        assert hasattr(criteria, "__iter__"), criteria.__class__.__name__
        assert isinstance(msg, str), msg.__class__.__name__
        assert isinstance(inplace, bool), inplace.__class__.__name__
        criteria = torch.squeeze(torch.asarray(criteria))
        assert criteria.ndim == 1, f"only a 1d vector is accepted, shape is {criteria.shape}"
        if criteria.dtype is torch.bool:  # case mask -> convert into index list
            assert criteria.shape[0] == len(self), (
                "the mask has to have the same length as the number of spots, "
                f"there are {len(self)} spots and mask is of len {criteria.shape[0]}"
            )
            criteria = torch.arange(len(self), dtype=torch.int64, device=criteria.device)[criteria]
        elif criteria.dtype != torch.int64:
            criteria = criteria.to(torch.int64)

        # manage inplace
        nb_spots = len(self)  # flush in background
        if not inplace:
            self = self.clone()  # pylint: disable=W0642

        # update history, it has to be done before changing state to be catched by signature
        self._history.append(f"{nb_spots} to {len(criteria)} spots: {msg}")

        # apply filter
        with self._rois_lock:
            self._rois = filter_by_indices(criteria, *self._rois)

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

    def get_property(self, name: str) -> object:
        """Return the property associated to te given id.

        Parameters
        ----------
        name : str
            The name of the property to get.

        Returns
        -------
        property : object
            The property value set with ``add_property``.

        Raises
        ------
        KeyError
            Is the property has never been defined or if the state changed.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.add_property("prop1", value="any python object 1", erasable=False)
        >>> diagram.add_property("prop2", value="any python object 2")
        >>> diagram.get_property("prop1")
        'any python object 1'
        >>> diagram.get_property("prop2")
        'any python object 2'
        >>> diagram.find_spots()  # change state
        >>> diagram.get_property("prop1")
        'any python object 1'
        >>> try:
        ...     diagram.get_property("prop2")
        ... except KeyError as err:
        ...     print(err)
        ...
        "the property 'prop2' is no longer valid because the state of the diagram has changed"
        >>> try:
        ...     diagram.get_property("prop3")
        ... except KeyError as err:
        ...     print(err)
        ...
        "the property 'prop3' does no exist"
        >>>
        """
        assert isinstance(name, str), name.__class__.__name__
        with self._cache[0]:
            try:
                state, value = self._properties[name]
            except KeyError as err:
                raise KeyError(f"the property {repr(name)} does no exist") from err
        if state is not None and state != self.state:
            # with self._lock:
            #     self._properties[name] = (state, None)
            raise KeyError(
                f"the property {repr(name)} is no longer valid "
                "because the state of the diagram has changed"
            )
        return value

    @property
    def history(self) -> list[str]:
        """Return the actions performed on the Diagram since the initialisation."""
        if not self.is_init():
            return []
        self.flush()
        return self._history.copy()  # copy for user protection

    def is_init(self) -> bool:
        """Return True if the diagram has been initialized.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.is_init()
        False
        >>> diagram.find_spots()
        >>> diagram.is_init()
        True
        >>> diagram.flush()
        >>> diagram.is_init()
        True
        >>>
        """
        return self._rois is not None or self._find_spots_kwargs is not None

    @property
    def image(self) -> torch.Tensor:
        """Return the complete image of the diagram.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.image.shape
        torch.Size([2018, 2016])
        >>> diagram.image.min() >= 0
        tensor(True)
        >>> diagram.image.max() <= 1
        tensor(True)
        >>>
        """
        with self._cache[0]:
            if "image" not in self._cache[1]:  # no auto ache because it is state invariant
                self._cache[1]["image"] = (
                    read_image(self._file_or_data)
                    if isinstance(self._file_or_data, pathlib.Path) else
                    self._file_or_data
                )
            return self._cache[1]["image"]

    def plot(
        self,
        disp=None,
        vmin: typing.Optional[numbers.Real] = None,
        vmax: typing.Optional[numbers.Real] = None,
        **kwargs,
    ):
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
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

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
                image.numpy(force=True),
                aspect="equal",
                extent=(.5, self.image.shape[1]+.5, self.image.shape[0]+.5, .5),
                interpolation=None,  # antialiasing is True
                norm="log",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
        if kwargs.get("show_boxes", True) and len(self):
            bboxes = self.bboxes.numpy(force=True).astype(np.float32)
            bboxes[:, :2] += 0.5  # ok inplace because copy has been made at previous line
            axes.plot(
                np.vstack((
                    bboxes[:, 1],
                    bboxes[:, 1],
                    bboxes[:, 1]+bboxes[:, 3],
                    bboxes[:, 1]+bboxes[:, 3],
                    bboxes[:, 1],
                )),
                np.vstack((
                    bboxes[:, 0],
                    bboxes[:, 0]+bboxes[:, 2],
                    bboxes[:, 0]+bboxes[:, 2],
                    bboxes[:, 0],
                    bboxes[:, 0],
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
        self.flush()
        with self._rois_lock:
            _, bboxes = self._rois
        return rawshapes2rois(imgbboxes2raw(self.image, bboxes), bboxes[:, 2:])

    @property
    def rois(self) -> typing.Union[None, torch.Tensor]:
        """Return the tensor of the provided rois of the spots.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.find_spots()
        >>> diagram.rawrois.shape
        torch.Size([219, 69, 19])
        >>> diagram.rois.shape
        torch.Size([219, 69, 19])
        >>> diagram.rois.mean() < diagram.rawrois.mean()  # no background
        tensor(True)
        >>>
        """
        if not self.is_init():
            return None
        self.flush()
        with self._rois_lock:
            datarois, bboxes = self._rois
        return rawshapes2rois(datarois, bboxes[:, 2:])

    def set_spots(self, new_spots: typing.Container) -> None:
        """Set the new spots as the current spots, reset the history and the cache.

        Paremeters
        ----------
        new_spots : typing.Container
            * Can be an iterable (list, tuple, set, ...).
            * Can be an arraylike of bounding boxes n * (anchor_i, anchor_j, height, width)
            * Can be a tuple of anchors (arraylike of shape n * (anchor_i, anchor_i))
                and rois (iterable of images patches).
            * Can be None for reset all (same as calling the `reset` method)
        """
        assert hasattr(new_spots, "__iter__"), new_spots.__class__.__name__

        # clear history and internal state (self._rois is cleared later)
        with self._cache[0]:
            self._cache[1].clear()
        self._history = []

        # case tensor
        if isinstance(new_spots, np.ndarray):
            new_spots = new_spots.from_numpy(new_spots)
        if isinstance(new_spots, torch.Tensor):
            if new_spots.dtype.is_floating_point:
                new_spots = new_spots.to(torch.int16)
            if new_spots.ndim == 2 and new_spots.shape[1] == 4:  # case from bounding boxes
                return self._set_spots_from_bboxes(new_spots)
            raise NotImplementedError(
                f"impossible to set new spots from an array of shape {new_spots.shape}, "
                "if has to be of shape (n, 4) for bounding boxes"
            )

        # preparation of data
        if not isinstance(new_spots, list):
            new_spots = list(new_spots)  # freeze the order

        # case from anchors, rois
        if (
            len(new_spots) == 2
            and isinstance(new_spots[0], (list, tuple, np.ndarray, torch.Tensor))
            and isinstance(new_spots[1], (list, tuple))
            and len(new_spots[0]) == len(new_spots[1])
        ):
            if not isinstance(new_spots[0], torch.Tensor):  # no torch.as_array memory leak
                new_spots[0] = torch.asarray(new_spots[0], dtype=torch.int16)
            anchors = new_spots[0]
            rois = [to_floattensor(roi) for roi in new_spots[1]]
            return self._set_spots_from_anchors_rois(anchors, rois)

        # case tensor (recursive delegation)
        cls = {s.__class__ for s in new_spots}
        if len(cls) == 1 and issubclass(next(iter(cls)), (list, tuple)):
            return self.set_spots(torch.tensor(new_spots, dtype=torch.int16))

        raise NotImplementedError(
            f"impossible to set new spots from {new_spots}, "
            "it has to be a container of Spot instance "
            "or a container of bounding boxes"
        )

    @property
    def state(self) -> str:
        """Return a hash of the diagram.

        If two diagrams gots the same state, it means they are the same.
        The hash take in consideration the internal state of the diagram.
        The retruned value is a hexadecimal string of length 32.

        Examples
        --------
        >>> from laueimproc.classes.base_diagram import BaseDiagram
        >>> from laueimproc.io import get_sample
        >>> diagram = BaseDiagram(get_sample())
        >>> diagram.state
        '3e50c54ac44f75f23d8f5c3170d5ecfb'
        >>> diagram.find_spots()
        >>> diagram.state
        '58291975dd11f34db66378d50e8a87f1'
        >>>
        """
        hasher = hashlib.md5(usedforsecurity=False)
        if isinstance(self._file_or_data, pathlib.Path):
            hasher.update(str(self._file_or_data.name).encode())
        else:
            hasher.update(id(self._file_or_data).to_bytes(8, "big"))
        hasher.update(str(self._find_spots_kwargs).encode())  # order same since python3.7
        hasher.update("\n".join(self._history[1:]).encode())
        return hasher.hexdigest()
