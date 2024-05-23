#!/usr/bin/env python3

"""Define the pytonic structure of a basic BaseDiagramsDataset."""

import hashlib
import inspect
import multiprocessing.pool
import numbers
import pathlib
import queue
import re
import threading
import time
import traceback
import typing
import warnings

import cloudpickle
import numpy as np
import psutil
import torch
import tqdm

from laueimproc.common import time2sec
from .diagram import Diagram


NCPU = len(psutil.Process().cpu_affinity())


def _excepthook(args):
    """Raise the exceptions comming from a BaseDiagramsDataset thread."""
    if isinstance(args.thread, (BaseDiagramsDataset, _ChainThread)):
        traceback.print_tb(args.exc_traceback)
        raise args.exc_value


threading.excepthook = _excepthook


def default_diag2ind(diagram: Diagram) -> int:
    """General function to find the index of a diagram from the filename.

    Parameters
    ----------
    diagram : laueimproc.classes.diagram.Diagram
        The diagram to index.

    Returns
    -------
    index : int
        The int index written in the file name.
    """
    assert isinstance(diagram, Diagram), diagram.__class__.__name__
    if diagram.file is None:
        raise ValueError(
            f"impossible to find the file based index of the diagram {repr(diagram)}, "
            "please provide the function `diag2ind`"
        )
    if not (candidates := re.findall(r"\d+", diagram.file.stem)):
        raise ValueError(
            f"failed to extract the index from the filename of the diagram {repr(diagram)}, "
            "please provide the function `diag2ind`"
        )
    return int(candidates[-1])


class _ChainThread(threading.Thread):
    """Heper for parallel real time filter chain."""

    def __init__(self, diagram: Diagram, chain: list, rescontainer: queue.Queue):
        self.diagram = diagram
        self.chain = chain.copy()
        self.rescontainer = rescontainer
        super().__init__(daemon=True)

    def run(self):
        """Apply all the operation on the diagram."""
        for func, args in self.chain:
            func(self.diagram, *args)
        self.rescontainer.put((self.diagram, len(self.chain)))


class BaseDiagramsDataset(threading.Thread):
    """A Basic diagrams dataset with the fondamental structure.

    Attributes
    ----------
    indices : set[int]
        The diagram indices currently reachable in the dataset.
    """

    def __init__(
        self,
        *diagram_refs,
        diag2ind: typing.Optional[typing.Callable[[Diagram], numbers.Integral]] = None,
        position: typing.Optional[typing.Callable[[int], tuple[numbers.Real, numbers.Real]]] = None,
    ):
        """Initialise the dataset.

        Parameters
        ----------
        *diagram_refs : tuple
            The diagram references, transmitted to ``add_diagrams``.
        diag2ind : callable, default=laueimproc.classses.dataset.default_diag2ind
            The function to associate an index to a diagram.
            If provided, this function has to be pickleable.
            It has to take only one positional argument (a simple Diagram instance)
            and to return a positive integer.
        position : callable, optional
            Provides information on the phisical position of the images on the sample.
            This function takes as input the index of the diagram,
            and returns the 2 coordinates in the sample plane.
        """
        if diag2ind is None:
            self._diag2ind = default_diag2ind
        else:
            self._diag2ind = cloudpickle.loads(self._check_diag2ind(diag2ind))
        if position is not None:
            position = cloudpickle.loads(self._check_position(position))
        self._position = [position, {}]  # the dict associate (x, y) to index
        self._diagrams: dict[int, typing.Union[Diagram, queue.Queue]] = {}  # index to diagram
        self._lock = threading.Lock()
        self._async_job: dict = {"dirs": [], "readed": set()}  # for the async run method
        self._operations_chain: list[tuple[typing.Callable[[Diagram], object], tuple]] = []
        self._properties: dict[str, tuple[typing.Union[None, str], object]] = {}  # the properties
        super().__init__(daemon=True)
        self.add_diagrams(diagram_refs)

    def __getstate__(self):
        """Make the dataset pickleable.

        Notes
        -----
        Only copy the ready diagrams.
        No queue left in _diagrams dictionary.
        """
        self.flush()
        with self._lock:
            pos_data = self._position[1]
            return (
                cloudpickle.dumps(self._diag2ind),
                [
                    cloudpickle.dumps(self._position[0]),
                    (pos_data.copy() if isinstance(pos_data, dict) else pos_data),
                ],
                {i: d for i, d in self._diagrams.items() if isinstance(d, Diagram)},
                cloudpickle.dumps(self._operations_chain),
                self._properties.copy(),
            )

    def __getitem__(self, item: numbers.Integral):
        """Get a diagram or a subset of the set.

        Parameters
        ----------
        item : int or slice
            The item of the diagram you want to reach.

        Returns
        -------
        Diagram or BaseBaseDiagramsDataset
            If the item is integer, return the corresponding diagram,
            return a frozen sub dataset view if it is a slice.

        Raises
        ------
        IndexError
            If the diagram of index `item` is not founded yet.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>>
        >>> # from integer
        >>> dataset[0]
        Diagram(img_00.jp2)
        >>> dataset[10]
        Diagram(img_10.jp2)
        >>> dataset[-10]
        Diagram(img_90.jp2)
        >>>
        >>> # from slice
        >>> subdataset = dataset[:30:10]
        >>> sorted(subdataset, key=str)
        [Diagram(img_00.jp2), Diagram(img_10.jp2), Diagram(img_20.jp2)]
        >>>
        >>> # from indices
        >>> subdataset = dataset[[0, 10, -10]]
        >>> sorted(subdataset, key=str)
        [Diagram(img_00.jp2), Diagram(img_10.jp2), Diagram(img_90.jp2)]
        >>>
        >>> # from condition
        >>> prop = torch.full((100,), False, dtype=bool)
        >>> prop[[0, 10, -10]] = True
        >>> subdataset = dataset[prop]
        >>> sorted(subdataset, key=str)
        [Diagram(img_00.jp2), Diagram(img_10.jp2), Diagram(img_90.jp2)]
        >>>
        """
        if isinstance(item, numbers.Integral):
            return self._get_diagram_from_index(item)
        if isinstance(item, slice):
            item = range(*item.indices(max(self._diagrams)+1))
        if isinstance(item, range):  # faster than torch.asarray
            item = np.fromiter(item, dtype=np.int64)
        if isinstance(item, (list, np.ndarray, torch.Tensor)):
            item = torch.asarray(item).reshape(-1)
            if item.dtype is torch.bool:
                item = torch.arange(len(item), dtype=torch.int64, device=item.device)[item]
            elif item.dtype != torch.int64:
                item = item.to(torch.int64)
            return self._get_diagrams_from_indices(item)
        raise TypeError(f"only int and slices are allowed, not {item}")

    def __iter__(self) -> Diagram:
        """Yield the diagrams ready.

        * This function is dynamic in the sense that if a new diagram is poping up
        during iterating, it's gotta be yield as well.
        * Diagrams are iterated in an arbitray, non-repeatable order

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> count = 0
        >>> for diagram in dataset:
        ...     count += 1
        ...
        >>> count
        100
        >>>
        """
        yielded: set[int] = set()  # the yielded diagram indices
        while True:
            with self._lock:
                if not (toyield := self._diagrams.keys() - yielded):
                    break
            for index in toyield:
                try:
                    diag = self._get_diagram_from_index(index)
                except IndexError:
                    pass
                yielded.add(index)
                yield diag

    def __len__(self) -> int:
        """Return the numbers of diagrams currently present in the dataset.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> len(BaseDiagramsDataset(get_samples()))
        100
        >>>
        """
        return len(self._diagrams)

    def __repr__(self) -> str:
        """Give a very compact representation.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> BaseDiagramsDataset(get_samples())
        <BaseDiagramsDataset with 100 diagrams>
        >>>
        """
        return f"<{self.__class__.__name__} with {len(self)} diagrams>"

    def __setstate__(self, state: tuple):
        """Fill the internal attributes.

        Usefull for pickle.

        Notes
        -----
        * No verification is made because the user is not supposed to call this method.

        Examples
        --------
        >>> import pickle
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> dataset_bis = pickle.loads(pickle.dumps(dataset))
        >>> assert id(dataset) != id(dataset_bis)
        >>> assert dataset.state == dataset_bis.state
        >>>
        """
        # not verification for thread safe
        # because this method is never meant to be call from a thread.
        (
            diag2ind_ser,
            self._position,
            self._diagrams,
            operations_chain_ser,
            self._properties,
        ) = state
        self._diag2ind = cloudpickle.loads(diag2ind_ser)
        self._position[0] = cloudpickle.loads(self._position[0])
        self._operations_chain = cloudpickle.loads(operations_chain_ser)
        self._lock = threading.Lock()
        self._async_job: dict = {"dirs": [], "readed": set()}  # for the async run method
        super().__init__(daemon=True)

    def __str__(self) -> str:
        """Return an exaustive printable string giving informations on the dataset.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> print(BaseDiagramsDataset(get_samples()))  # doctest: +ELLIPSIS
        BaseDiagramsDataset from the folder /home/.../.cache/laueimproc/samples:
            No function has been applied.
            No Properties
            Current state:
                * id, state: ...
                * nbr diags: 100
                * 2d indexing: no
        >>>
        """
        # title
        if len(folders := {d.file.parent for d in self if d.file is not None}) == 1:
            text = f"BaseDiagramsDataset from the folder {folders.pop()}:"
        else:
            text = "BaseDiagramsDataset:"

        with self._lock:
            # history
            if self._operations_chain:
                text += "\n    Function chain:"
                for i, (func, args) in enumerate(self._operations_chain):
                    args = ", ".join(("diag",) + tuple(repr(a) for a in args))
                    text += f"\n        {i+1}: [{func.__name__}({args}) for diag in self]"
            else:
                text += "\n    No function has been applied."

            # properties
            if self._properties:
                text += "\n    Properties:"
                for name, (_, value) in self._properties.items():
                    if len(value_str := str(value).replace("\n", "\\n")) >= 80:
                        value_str = f"<{value.__class__.__name__} object>"
                    text += f"\n        * {name}: {value_str}"
            else:
                text += "\n    No Properties"

            # stats
            text += "\n    Current state:"
            text += f"\n        * id, state: {id(self)}, {self.state}"
            text += f"\n        * nbr diags: {len(self)}"
            if self._position[0] is None:
                text += "\n        * 2d indexing: no"
            else:
                text += f"\n        * 2d indexing: {self._position[0]}"

        return text

    @staticmethod
    def _check_apply(func: typing.Callable[[Diagram], object]) -> bytes:
        """Ensure that the function has right input / output."""
        assert callable(func), f"`func` has to be callable, not {func.__class__.__name__}"
        try:
            serfunc = cloudpickle.dumps(func)
        except TypeError as err:
            raise AssertionError(f"the function `func` {func} is not pickleable") from err
        signature = inspect.signature(func)
        assert len(signature.parameters) >= 1, \
            "the function `func` has to take at least 1 parameter"
        parameter = next(iter(signature.parameters.values()))
        assert parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        if parameter.annotation is parameter.empty:
            warnings.warn("please specify the input type of `func`", SyntaxWarning)
        elif parameter.annotation is not Diagram:
            raise AssertionError(
                f"the function `func` has to get a Diagram, not {parameter.annotation}"
            )
        return serfunc

    @staticmethod
    def _check_diag2ind(diag2ind: typing.Callable[[Diagram], numbers.Integral]) -> bytes:
        """Ensure that the indexing function has right input / output."""
        assert callable(diag2ind), \
            f"`diag2ind` has to be callable, not {diag2ind.__class__.__name__}"
        try:
            serfunc = cloudpickle.dumps(diag2ind)
        except TypeError as err:
            raise AssertionError(f"the function `diag2ind` {diag2ind} is not picklable") from err
        signature = inspect.signature(diag2ind)
        assert len(signature.parameters) == 1, "the function `diag2ind` has to take 1 parameter"
        parameter = next(iter(signature.parameters.values()))
        assert parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        if parameter.annotation is parameter.empty:
            warnings.warn("please specify the input type of `diag2ind`", SyntaxWarning)
        elif parameter.annotation is not Diagram:
            raise AssertionError(
                f"the function `diag2ind` has to get a Diagram, not {parameter.annotation}"
            )
        if signature.return_annotation is parameter.empty:
            warnings.warn("please specify the return type of `diag2ind`", SyntaxWarning)
        elif not issubclass(signature.return_annotation, numbers.Integral):
            raise AssertionError(
                f"the function `diag2ind` has to return int, not {signature.return_annotation}"
            )
        return serfunc

    @staticmethod
    def _check_position(
        position: typing.Callable[[int], tuple[numbers.Real, numbers.Real]]
    ) -> bytes:
        """Ensure that the position function has the right input / outputs."""
        assert callable(position), \
            f"`position` has to be callable, not {position.__class__.__name__}"
        try:
            serfunc = cloudpickle.dumps(position)
        except TypeError as err:
            raise AssertionError(f"the function `position` {position} is not picklable") from err
        signature = inspect.signature(position)
        assert len(signature.parameters) == 1, "the function `position` has to take 1 parameter"
        parameter = next(iter(signature.parameters.values()))
        if parameter.annotation is parameter.empty:
            warnings.warn("please specify the input type of `position`", SyntaxWarning)
        elif parameter.annotation is not int:
            raise AssertionError(
                f"the function `position` has to get a int, not {parameter.annotation}"
            )
        if signature.return_annotation is parameter.empty:
            warnings.warn("please specify the return type of `position`", SyntaxWarning)
        elif (
            signature.return_annotation is not tuple
            and typing.get_origin(signature.return_annotation) is not tuple
        ):
            raise AssertionError(
                f"the function `position` has to return tuple, not {signature.return_annotation}"
            )
        return_args = typing.get_args(signature.return_annotation)
        assert len(return_args) == 2, \
            f"the function `position` has to return 2 elements, not {len(return_args)}"
        assert issubclass(return_args[0], numbers.Real), \
            f"first output element of `position` has to be a real number, not a {return_args[0]}"
        assert issubclass(return_args[1], numbers.Real), \
            f"second output element of `position` has to be a real number, not a {return_args[1]}"
        return serfunc

    def _get_diagram_from_coord(self, first_idx: numbers.Real, second_idx: numbers.Real) -> Diagram:
        """Return the closest diagram of the given position."""
        assert isinstance(first_idx, numbers.Real), first_idx.__class__.__name__
        assert isinstance(second_idx, numbers.Real), second_idx.__class__.__name__
        with self._lock:
            if isinstance(self._position[1], dict):  # dict to tensor
                coords, indices = zip(*self._position[1].items())
                self._position[1] = (
                    torch.tensor(coords, dtype=torch.float32),
                    torch.tensor(indices, dtype=torch.int64),
                )
            coord = torch.tensor(
                [[first_idx, second_idx]],
                dtype=self._position[1][0].dtype,
                device=self._position[1][0].device,
            )
            dist = torch.sum((self._position[1][0].unsqueeze(0) - coord.unsqueeze(1))**2, dim=2)
            pos = torch.argmin(dist, dim=1).item()
            index = self._position[1][1][pos].item()
        return self._get_diagram_from_index(index)

    def _get_diagram_from_index(self, index: numbers.Integral) -> Diagram:
        """Return the diagram of index `index`."""
        assert isinstance(index, numbers.Integral), index.__class__.__name__
        if index < 0:
            index += len(self)
            assert index >= 0, "the provided index is too negative"
        index = int(index)
        with self._lock:
            try:
                diagram = self._diagrams[index]
            except KeyError as err:
                raise IndexError(f"The diagram of index {index} is not in the dataset") from err
            if isinstance(diagram, queue.Queue):
                diagram, nb_ops = diagram.get()  # passive waiting
                for func, args in self._operations_chain[nb_ops:]:  # simple precaution
                    func(diagram, *args)
                self._diagrams[index] = diagram
        return diagram

    def _get_diagrams_from_indices(self, indices: torch.Tensor):
        """Return a frozen sub dataset view."""
        if (isneg := indices < 0).any():
            indices = indices.clone()
            with self._lock:
                corrected_indices = indices[isneg] + len(self)
            assert (corrected_indices >= 0).all(), "some provided indices are too negative"
            indices[isneg] = corrected_indices
        diagrams = {i: self._get_diagram_from_index(i) for i in indices.tolist()}
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.__setstate__(self.__getstate__())
        new_dataset._diagrams = diagrams  # pylint: disable=W0212
        return new_dataset

    def add_property(self, name: str, value: object, *, erasable: bool = True):
        """Add a property to the dataset.

        Parameters
        ----------
        name : str
            The identifiant of the property for the requests.
            If the property is already defined with the same name, the new one erase the older one.
        value
            The property value. If a number is provided, it will be faster.
        erasable : boolean, default=True
            If set to False, the property will be set in stone,
            overwise, the property will desappear as soon as the dataset state changed.
        """
        assert isinstance(name, str), name.__class__.__name__
        assert isinstance(erasable, bool), erasable.__class__.__name__
        with self._lock:
            self._properties[name] = ((self.state if erasable else None), value)

    def add_diagram(self, new_diagram: Diagram):
        """Append a new diagram into the dataset.

        Parameters
        ----------
        new_diagram : laueimproc.classes.diagram.Diagram
            The new instanciated diagram not already present in the dataset.

        Raises
        ------
        LookupError
            If the diagram is already present in the dataset.
        """
        assert isinstance(new_diagram, Diagram), new_diagram.__class__.__name__

        # get and check index
        index = self._diag2ind(new_diagram)
        assert isinstance(index, numbers.Integral), (
            f"the function {self._diag2ind} must return an integer, "
            f"not a {index.__class__.__name__}"
        )
        assert index >= 0, \
            f"the function {self._diag2ind} must return a positive number, not {index}"
        index = int(index)

        # get an check sample position
        if self._position[0] is not None:
            coord = self._position[0](index)
            assert isinstance(coord, tuple), \
                f"the function {self._position[0]} must return a tuple, not {coord}"
            assert len(coord) == 2, \
                f"the function {self._position[0]} must return 2 elements, not {len(coord)}"
            assert isinstance(coord[0], numbers.Real), \
                f"the first element returned by {self._position[0]} must be a number not {coord[0]}"
            assert isinstance(coord[1], numbers.Real), \
                f"the first element returned by {self._position[1]} must be a number not {coord[1]}"
            coord = (float(coord[0]), float(coord[1]))
        else:
            coord = None

        # add diagram and check unicity
        with self._lock:
            if index in self._diagrams:
                raise LookupError(
                    f"the index {index} of the diagram {new_diagram} is already in the dataset"
                )
            if coord is not None:  # add diagram position
                if isinstance(self._position[1], tuple):  # tensor to dict
                    self._position[1] = {
                        (x, y): i for (x, y), i in
                        zip(self._position[1][0].tolist(), self._position[1][1].tolist())
                    }
                self._position[1][coord] = index
            if new_diagram.file is not None:
                self._async_job["readed"].add(new_diagram.file)
            if self._operations_chain:
                self._diagrams[index] = queue.Queue()
                thread = _ChainThread(new_diagram, self._operations_chain, self._diagrams[index])
            else:
                self._diagrams[index] = new_diagram
                thread = None
        if thread is not None:  # starts the thread (apply the function chain to the new diagram)
            if len([None for t in threading.enumerate() if isinstance(t, _ChainThread)]) < 3*NCPU:
                thread.start()  # asnyc
                if not self._started.is_set():
                    self.start()
            else:
                thread.run()  # blocking

    def add_diagrams(
        self, new_diagrams: typing.Union[typing.Iterable, Diagram, str, pathlib.Path]
    ):
        """Append the new diagrams into the datset.

        Parameters
        ----------
        new_diagrams
            The diagram references, they can be of this natures:

                * laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.
                * iterable : An iterable of any of the types specified above.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_samples
        >>> file = min(get_samples().iterdir())
        >>>
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams(Diagram(file))  # from Diagram instance
        >>> dataset
        <BaseDiagramsDataset with 1 diagrams>
        >>>
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams(file)  # from filename (pathlib)
        >>> dataset
        <BaseDiagramsDataset with 1 diagrams>
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams(str(file))  # from filename (str)
        >>> dataset
        <BaseDiagramsDataset with 1 diagrams>
        >>>
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams(get_samples())  # from folder (pathlib)
        >>> dataset
        <BaseDiagramsDataset with 100 diagrams>
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams(str(get_samples()))  # from folder (str)
        >>> dataset
        <BaseDiagramsDataset with 100 diagrams>
        >>>
        >>> # from iterable (DiagramDataset, list, tuple, ...)
        >>> # ok with nested iterables
        >>> dataset = BaseDiagramsDataset()
        >>> dataset.add_diagrams([Diagram(f) for f in get_samples().iterdir()])
        >>> dataset
        <BaseDiagramsDataset with 100 diagrams>
        >>>
        """
        if isinstance(new_diagrams, Diagram):
            self.add_diagram(new_diagrams)
        elif isinstance(new_diagrams, pathlib.Path):
            new_diagrams.expanduser().resolve()
            assert new_diagrams.exists(), f"{new_diagrams} if not an existing path"
            if new_diagrams.is_dir():
                self.add_diagrams(  # optional, no procrastination
                    f for f in new_diagrams.iterdir()
                    if f.suffix.lower() in {".jp2", ".mccd", ".tif", ".tiff"}
                )
                self._async_job["dirs"].append([new_diagrams, 0.0])
                if not self._started.is_set():
                    self.start()
            else:
                self.add_diagram(Diagram(new_diagrams))  # case file
        elif isinstance(new_diagrams, (str, bytes)):
            self.add_diagrams(pathlib.Path(new_diagrams))
        elif isinstance(
            new_diagrams, (tuple, list, set, frozenset, typing.Generator, self.__class__)
        ):
            for new_diagram in new_diagrams:
                self.add_diagrams(new_diagram)
        else:
            raise ValueError(f"the `new_diagrams` {new_diagrams} are not recognised")

    def apply(
        self,
        func: typing.Callable[[Diagram], object],
        args: typing.Optional[tuple] = None,
        kwargs: typing.Optional[dict] = None,
    ) -> dict[int, object]:
        """Apply an operation in all the diagrams of the dataset.

        Parameters
        ----------
        func : callable
            A function that take a diagram and optionaly other parameters, and return anything.
            The function can modify the diagram inplace. It has to be pickaleable.
        args : tuple, optional
            Positional arguments transmitted to the provided func.
        kwargs : dict, optional
            Keyword arguments transmitted to the provided func.

        Returns
        -------
        res : dict
            The result of the function for each diagrams.
            To each diagram index, associate the result.

        Notes
        -----
        This function will be automaticaly applided on the new diagrams, but the result is throw.

        Examples
        --------
        >>> from pprint import pprint
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.classes.diagram import Diagram
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())[::10]  # subset to go faster
        >>> def peak_search(diagram: Diagram, density: float) -> int:
        ...     '''Return the number of spots.'''
        ...     diagram.find_spots(density=density)
        ...     return len(diagram)
        ...
        >>> res = dataset.apply(peak_search, args=(0.5,))
        >>> pprint(res)
        {0: 204,
         10: 547,
         20: 551,
         30: 477,
         40: 404,
         50: 537,
         60: 2121,
         70: 274,
         80: 271,
         90: 481}
        >>>
        """
        # verifications
        func = cloudpickle.loads(self._check_apply(func))
        if args is None:
            args = ()
        else:
            assert isinstance(args, tuple), args.__class__.__name__
            try:
                cloudpickle.dumps(args)
            except TypeError as err:
                raise AssertionError(
                    f"the argument `args` {args} is not serializable"
                ) from err
        if kwargs is None:
            kwargs = {}
        else:
            assert isinstance(kwargs, dict), kwargs.__class__.__name__
            assert all(isinstance(k, str) for k in kwargs), kwargs
            try:
                cloudpickle.dumps(kwargs)
            except TypeError as err:
                raise AssertionError(
                    f"the argument `kwargs` {kwargs} is not serializable"
                ) from err

        # apply
        idxs_diags = [(i, self._get_diagram_from_index(i)) for i in sorted(self.indices)]  # frozen
        with self._lock, multiprocessing.pool.ThreadPool(NCPU) as pool:
            res = dict(
                tqdm.tqdm(
                    pool.imap_unordered(
                        lambda idx_diag: (idx_diag[0], func(idx_diag[1], *args, **kwargs)),
                        idxs_diags,
                    ),
                    desc=func.__name__,
                    unit="diag",
                    total=len(idxs_diags),
                )
            )
            self._operations_chain.append((func, args))

        return res

    def autosave(
        self,
        filename: typing.Union[str, pathlib.Path],
        restore: bool = True,
        delay: typing.Union[numbers.Real, str] = None,
    ) -> pathlib.Path:
        """Manage the dataset recovery.

        This allows the dataset to be backuped at regular time intervals,
        making it possible to restore where the diagram has been left up.

        Parameters
        ----------
        filename : pathlike
            The name of the persistant file,
            transmitted to ``laueimproc.io.save_dataset.write_dataset``.
        restore : boolean, default=True
            If True and the file exists, it update the dataset with the recorded state.
            If set to False, the file is never readed, written only.
        delay : float or str, optional
            If provided and > 0, automatic checkpoint will be performed.
            it corresponds to the time interval between two check points.
            The supported formats are defined in ``laueimproc.common.time2sec``.

        Returns
        -------
        filename : pathlib.Path
            The real absolute path.
        """
        from laueimproc.io.save_dataset import restore_dataset, write_dataset

        assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
        filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".pickle")
        assert isinstance(restore, bool), restore.__class__.__name__

        if restore and filename.exists():
            restore_dataset(filename, self)
        else:
            filename = write_dataset(filename, self)
        if not delay:
            if "autosave" in self._async_job:
                del self._async_job["autosave"]
        else:
            delay = time2sec(delay)
            self._async_job["autosave"] = [delay, time.time(), filename]
            if not self._started.is_set():
                self.start()

        return filename

    def clone(self, **kwargs):
        """Instanciate a new identical dataset.

        Parameters
        ----------
        **kwargs : dict
            Transmitted to ``laueimproc.classes.base_diagram.BaseDiagram.clone``.

        Returns
        -------
        BaseDiagramsDataset
            The new copy of self.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> dataset_bis = dataset.clone()
        >>> assert id(dataset) != id(dataset_bis)
        >>> assert dataset.state == dataset_bis.state
        >>>
        """
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.__setstate__(self.__getstate__())
        new_dataset._diagrams = {  # pylint: disable=W0212
            i: d.clone(**kwargs) for i, d in new_dataset._diagrams.items()  # pylint: disable=W0212
        }
        return new_dataset

    def flush(self):
        """Extract finished thread diagrams."""
        with self._lock:
            for index, diag_queue in self._diagrams.items():
                if isinstance(diag_queue, queue.Queue):
                    try:
                        diagram, nb_ops = diag_queue.get_nowait()
                    except queue.Empty:
                        continue
                    for func, args in self._operations_chain[nb_ops:]:  # simple precaution
                        func(diagram, *args)
                    self._diagrams[index] = diagram

    def get_property(self, name: str) -> object:
        """Return the property associated to the given id.

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
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> dataset.add_property("prop1", value="any python object 1", erasable=False)
        >>> dataset.add_property("prop2", value="any python object 2")
        >>> dataset.get_property("prop1")
        'any python object 1'
        >>> dataset.get_property("prop2")
        'any python object 2'
        >>> dataset = dataset[:1]  # change state
        >>> dataset.get_property("prop1")
        'any python object 1'
        >>> try:
        ...     dataset.get_property("prop2")
        ... except KeyError as err:
        ...     print(err)
        ...
        "the property 'prop2' is no longer valid because the state of the dataset has changed"
        >>> try:
        ...     dataset.get_property("prop3")
        ... except KeyError as err:
        ...     print(err)
        ...
        "the property 'prop3' does no exist"
        >>>
        """
        assert isinstance(name, str), name.__class__.__name__
        with self._lock:
            try:
                state, value = self._properties[name]
            except KeyError as err:
                raise KeyError(f"the property {repr(name)} does no exist") from err
        if state is not None and state != self.state:
            # with self._lock:
            #     self._properties[name] = (state, None)
            raise KeyError(
                f"the property {repr(name)} is no longer valid "
                "because the state of the dataset has changed"
            )
        return value

    @property
    def indices(self) -> set[int]:
        """Return the diagram indices currently reachable in the dataset.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> sorted(dataset[-10:10:-5].indices)
        [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        >>>
        """
        self.flush()
        with self._lock:
            return {i for i, d in self._diagrams.items() if isinstance(d, Diagram)}

    @property
    def state(self) -> str:
        """Return a hash of the dataset.

        If two datasets gots the same state, it means they are the same.
        The hash take in consideration the indices of the diagrams and the functions applyed.
        The retruned value is a hexadecimal string of length 32.

        Examples
        --------
        >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> dataset = BaseDiagramsDataset(get_samples())
        >>> dataset.state
        '047e5d6c00850898c233128e31e1f7e1'
        >>> dataset[:].state
        '047e5d6c00850898c233128e31e1f7e1'
        >>> dataset[:1].state
        '1de1605a297bafd22a886de7058cae81'
        >>>
        """
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(str(sorted(self._diagrams)).encode())
        hasher.update(cloudpickle.dumps(self._diag2ind))
        hasher.update(cloudpickle.dumps(self._position[0]))
        hasher.update(cloudpickle.dumps(self._operations_chain))
        return hasher.hexdigest()

    def run(self):
        """Run asynchronousely in a child thread, called by self.start()."""
        while True:
            # scan if a new diagram has arrived in the folder
            for directory_time in self._async_job["dirs"]:
                if directory_time[0].stat().st_mtime == directory_time[1]:
                    continue
                directory_time[1] = directory_time[0].stat().st_mtime
                for file in directory_time[0].iterdir():
                    if (
                        file in self._async_job["readed"]
                        or file.suffix.lower() not in {".jp2", ".mccd", ".tif", ".tiff"}
                    ):
                        continue
                    diagram = Diagram(file)
                    self.add_diagram(diagram)  # update self._async_job["readed"]

            # autosave
            if "autosave" in self._async_job:
                delay, last_backup, file = self._async_job["autosave"]
                if time.time() > last_backup + delay:
                    from laueimproc.io.save_dataset import write_dataset
                    write_dataset(file, self)
                    self._async_job["autosave"] = [delay, time.time(), file]

            time.sleep(10)
