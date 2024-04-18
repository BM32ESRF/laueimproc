#!/usr/bin/env python3

"""Link serveral diagrams together."""

import hashlib
import inspect
import multiprocessing.pool
import numbers
import pathlib
import pickle
import queue
import re
import threading
import time
import traceback
import typing
import warnings

import psutil
import torch
import tqdm

from .diagram import Diagram


GENTYPE = type((lambda: (yield))())
NCPU = len(psutil.Process().cpu_affinity())


def _excepthook(args):
    """Raise the exceptions comming from a DiagramDataset thread."""
    if isinstance(args.thread, (DiagramDataset, _ChainThread)):
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


class DiagramDataset(threading.Thread):
    """Link Diagrams together."""

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
            If provided, this function has to be pickalable.
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
            self._check_diag2ind(diag2ind)
            self._diag2ind = diag2ind
        if position is not None:
            self._check_position(position)
        self._position = [position, {}]  # the dict associate (x, y) to index
        self._diagrams: dict[int, Diagram] = {}  # index to diagram
        self._lock = threading.Lock()
        self._to_sniff: dict = {"dirs": [], "readed": set()}  # for the async run method
        self._operations_chain: list[tuple[typing.Callable[[Diagram], object], tuple]] = []
        self._properties: dict[str, tuple[typing.Union[None, str], object]] = {}  # the properties
        self.add_diagrams(diagram_refs)
        super().__init__(daemon=True)
        self.start()

    def __getitem__(self, item: numbers.Integral):
        """Get a diagram or a subset of the set.

        Parameters
        ----------
        item : int or slice
            The item of the diagram you want to reach.

        Returns
        -------
        Diagram or Dataset
            If the item is integer, return the corresponding diagram,
            return a frozen sub dataset view if it is a slice.

        Raises
        ------
        IndexError
            If the diagram of index `item` is not founded yet.
        """
        if isinstance(item, numbers.Integral):
            return self._get_diagram_from_index(item)
        if isinstance(item, slice):
            return self._get_diagrams_from_slice(item)
        if isinstance(item, tuple):  # case grid
            if self._position[0] is None:
                raise AttributeError(
                    "you must provide the `position` argument to acces the position"
                )
            assert len(item) == 2, f"only 2d position allow, not {len(item)}d"
            if isinstance(item[0], numbers.Real) and isinstance(item[1], numbers.Real):
                return self._get_diagram_from_coord(*item)

        raise TypeError(f"only int, slices and tuple[float, float] are allowed, not {item}")

    def __iter__(self) -> Diagram:
        """Yield the diagrams ready.

        This function is dynamic in the sense that if a new diagram is poping up
        during iterating, it's gotta be yield as well.
        """
        yielded = set()  # the yielded diagrams
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
        """Return the approximative numbers of diagrams soon be present in the dataset."""
        return len(self._diagrams)

    def __repr__(self) -> str:
        """Give a very compact representation."""
        return f"<{self.__class__.__name__} with {len(self)} diagrams of id {id(self)}>"

    def __str__(self) -> str:
        """Return an exaustive printable string giving informations on the dataset."""

        # title
        if len(folders := {d.file.parent for d in self if d.file is not None}) == 1:
            text = f"DiagramDataset from the folder {folders.pop()}:"
        else:
            text = "DiagramDataset:"

        with self._lock:
            # history
            if self._operations_chain:
                text += "\n    Function chain:"
                for i, func in enumerate(self._operations_chain):
                    text += f"\n        {i+1}: {func}"
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
            text += f"\n        * max index: {len(self)}"
            if self._position[0] is None:
                text += "\n        * 2d indexing: no"
            else:
                text += f"\n        * 2d indexing: {self._position[0]}"

        return text

    @staticmethod
    def _check_diag2ind(diag2ind: typing.Callable[[Diagram], numbers.Integral]):
        """Ensure that the indexing function has right input / output."""
        assert callable(diag2ind), \
            f"`diag2ind` has to be callable, not {diag2ind.__class__.__name__}"
        try:
            pickle.dumps(diag2ind)
        except pickle.PicklingError as err:
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

    @staticmethod
    def _check_position(position: typing.Callable[[int], tuple[numbers.Real, numbers.Real]]):
        """Ensures that the position function has the right input / outputs."""
        assert callable(position), \
            f"`position` has to be callable, not {position.__class__.__name__}"
        try:
            pickle.dumps(position)
        except pickle.PicklingError as err:
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

    def _get_diagram_from_coord(self, first_idx: numbers.Real, second_idx: numbers.Real) -> Diagram:
        """Return the closest diagram of the given position."""
        assert isinstance(first_idx, numbers.Real), first_idx.__class__.__name__
        assert isinstance(second_idx, numbers.Real), second_idx.__class__.__name__
        with self._lock:
            if isinstance(self._position[1], dict):  # dict to tensor
                coords, indexs = zip(*self._position[1].items())
                self._position[1] = (
                    torch.tensor(coords, dtype=torch.float32),
                    torch.tensor(indexs, dtype=int),
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
            assert index >= 0, "the provided index is to negative"
        index = int(index)
        with self._lock:
            try:
                diagram = self._diagrams[index]
            except KeyError as err:
                raise IndexError(f"The diagram of index {index} is not in the dataset") from err
            if isinstance(diagram, queue.Queue):
                diagram = diagram.get()  # passive waiting
                self._diagrams[index] = diagram
        return diagram

    def _get_diagrams_from_slice(self, indexs: slice):
        """Return a frozen sub dataset view."""
        assert isinstance(indexs, slice), indexs.__class__.__name__
        with self._lock:
            id_max = max(self._diagrams)
            indexs = range(*indexs.indices(id_max+1))
            try:
                diagrams = {i: self._diagrams[i] for i in indexs}
            except KeyError as err:
                raise IndexError("one of the asked diagram is not in the dataset") from err
            new_dataset = DiagramDataset()
            new_dataset._diag2ind = self._diag2ind
            new_dataset._position = self._position.copy()
            new_dataset._diagrams = diagrams
            new_dataset._operations_chain = self._operations_chain.copy()
            new_dataset._properties = self._properties.copy()
            return new_dataset

    def add_property(self, name: str, value: object, *, state_independent: bool = False):
        """Add a property to the dataset.

        Parameters
        ----------
        name : str
            The identifiant of the property for the requests.
            If the property is already defined with the same name, the new one erase the older one.
        value
            The property value. If a number is provided, it will be faster.
        state_independent : boolean, default=False
            If set to True, the property will be keep when filtering,
            overwise, the property will desappear as soon as the dataset state changed.
        """
        assert isinstance(name, str), name.__class__.__name__
        assert isinstance(state_independent, bool), state_independent.__class__.__name__
        with self._lock:
            self._properties[name] = ((None if state_independent else self.state), value)

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
                raise LookupError(f"the diagram of index {index} is already present in the dataset")
            if coord is not None:
                if isinstance(self._position[1], tuple):  # tensor to dict
                    self._position[1] = {
                        (x, y): i for (x, y), i in
                        zip(self._position[1][0].tolist(), self._position[1][1].tolist())
                    }
                self._position[1][coord] = index
            if new_diagram.file is not None:
                self._to_sniff["readed"].add(new_diagram.file)
            if self._operations_chain:
                self._diagrams[index] = queue.Queue()
                thread = _ChainThread(new_diagram, self._operations_chain, self._diagrams[index])
            else:
                self._diagrams[index] = new_diagram
                thread = None
        if thread is not None:
            if len([None for t in threading.enumerate() if isinstance(t, _ChainThread)]) < 3*NCPU:
                thread.start()  # asnyc
            else:
                thread.run()  # blocking

    def add_diagrams(
        self, new_diagrams: typing.Union[typing.Iterable, Diagram, str, bytes, pathlib.Path]
    ):
        """Append the new diagrams into the datset.

        Parameters
        ----------
        new_diagram
            The diagram references, they can be of this natures:

                * laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.
                * iterable : An iterable of any of the types specified above.
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
                self._to_sniff["dirs"].append([new_diagrams, 0.0])
            else:
                self.add_diagram(Diagram(new_diagrams))  # case file
        elif isinstance(new_diagrams, (str, bytes)):
            self.add_diagrams(pathlib.Path(new_diagrams))
        elif isinstance(new_diagrams, (tuple, list, set, frozenset, GENTYPE, self.__class__)):
            for new_diagram in new_diagrams:
                self.add_diagrams(new_diagram)
        else:
            raise ValueError(f"the `new_diagrams` {new_diagrams} are not recognised")

    def apply(
        self,
        func: typing.Callable[[Diagram], object],
        args: typing.Optional[tuple] = None,
    ) -> dict[int, object]:
        """Apply an operation in all the diagrams of the dataset.

        Parameters
        ----------
        func : callable
            A function that take a diagram and optionaly other parameters, and return anything.
            The function can modify the diagram inplace. It has to be pickaleable.
        args : tuple, optional
            Positional arguments to forward to the provided func.

        Returns
        -------
        res : dict
            The result of the function for each diagrams.
            To each diagram index, associate the result.

        Notes
        -----
        This function will be automaticaly applided on the new diagrams, but the result is throw.
        """
        # verifications
        assert callable(func), func.__class__.__name__
        try:
            pickle.dumps(func)
        except pickle.PicklingError as err:
            raise AssertionError(
                f"the function `func` {func} is not pickalable"
            ) from err
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
        if args is None:
            args = ()
        else:
            assert isinstance(args, tuple), args.__class__.__name__
            try:
                pickle.dumps(args)
            except pickle.PicklingError as err:
                raise AssertionError(
                    f"the arguments `args` {args} are not pickalable"
                ) from err

        # apply
        with self._lock, multiprocessing.pool.ThreadPool(NCPU) as pool:
            idxs_diags = list((i, d) for i, d in self._diagrams.items() if isinstance(d, Diagram))
            res = dict(
                tqdm.tqdm(
                    pool.imap_unordered(
                        lambda idx_diag: (idx_diag[0], func(idx_diag[1], *args)), idxs_diags
                    ),
                    desc=func.__name__,
                    unit="diag",
                    total=len(idxs_diags)
                )
            )
            self._operations_chain.append((func, args))

        # filter properties
        with self._lock:
            state = self.state
            for key in list(self._properties):
                match = self._properties[key][0]
                if match is not None and match != state:
                    del self._properties[key]

        return res

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
        """
        assert isinstance(name, str), name.__class__.__name__
        with self._lock:
            try:
                return self._properties[name][1]
            except KeyError as err:
                raise KeyError(f"the property {repr(name)} does no exist") from err

    @property
    def state(self) -> str:
        """Return a hash of the dataset.

        If two datasets gots the same state, it means they are the same.
        The hash take in consideration the indexes of the diagrams and the functions applyed.
        The retruned value is a hexadecimal string of length 32.
        """
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(str(sorted(self._diagrams)).encode())
        hasher.update(pickle.dumps(self._diag2ind))
        hasher.update(pickle.dumps(self._position[0]))
        hasher.update(pickle.dumps(self._operations_chain))
        return hasher.hexdigest()

    def run(self):
        """Run asynchronousely in a child thread, called by self.start()."""
        while True:
            # scan if a new diagram has arrived in the folder
            for directory_time in self._to_sniff["dirs"]:
                if directory_time[0].stat().st_mtime == directory_time[1]:
                    continue
                directory_time[1] = directory_time[0].stat().st_mtime
                for file in directory_time[0].iterdir():
                    if (
                        file in self._to_sniff["readed"]
                        or file.suffix.lower() not in {".jp2", ".mccd", ".tif", ".tiff"}
                    ):
                        continue
                    diagram = Diagram(file)
                    self.add_diagram(diagram)  # update self._to_sniff["readed"]

            # make accessible the just borned diagrams
            with self._lock:
                for idx in list(self._diagrams):  # copy for allowing modification
                    if not isinstance(self._diagrams[idx], queue.Queue):
                        continue
                    try:
                        diagram, nb_ops = self._diagrams[idx].get_nowait()
                    except queue.Empty:
                        continue
                    if nb_ops != len(self._operations_chain):  # if apply called during thread run
                        for func, args in self._operations_chain[nb_ops:]:
                            func(diagram, *args)
                    self._diagrams[idx] = diagram

            time.sleep(10)
