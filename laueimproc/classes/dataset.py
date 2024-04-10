#!/usr/bin/env python3

"""Link serveral diagrams together."""

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

import torch
import tqdm

from .diagram import Diagram


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
    """Link Diagrams together.

    Attributes
    ----------
    diagrams : list[Diagram]
        The full list of the diagrams. If a diagram is not in the dataset, it is squeeze.
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
        self._operations_chain: list[typing.Callable[[Diagram], object]] = []
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

    def __len__(self) -> int:
        """Return the approximative numbers of diagrams soon be present in the dataset."""
        return max(self._diagrams) + 1

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
        if not isinstance(diagram, Diagram):
            raise IndexError(f"The diagram of index {index} is coming soon in the dataset")
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
            new_dataset._operations_chain = self._operations_chain
            return new_dataset

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

        # get an check grid position
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
            if self._operations_chain:
                self._diagrams[index] = queue.Queue()
                _ChainThread(new_diagram, self._operations_chain, self._diagrams[index]).start()
            else:
                self._diagrams[index] = new_diagram

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
        gen_type = type((lambda: (yield))())
        if isinstance(new_diagrams, Diagram):
            self.add_diagram(new_diagrams)
        elif isinstance(new_diagrams, pathlib.Path):
            new_diagrams.expanduser().resolve()
            assert new_diagrams.exists(), f"{new_diagrams} if not an existing path"
            if new_diagrams.is_dir():
                self.add_diagrams(  # optional, no procrastination
                    (
                        f for f in new_diagrams.iterdir()
                        if f.suffix.lower() not in {".jp2", ".mccd", ".tif", ".tiff"}
                    )
                )
                self._to_sniff["dirs"].append([new_diagrams, 0.0])
            else:
                self.add_diagram(Diagram(new_diagrams))  # case file
        elif isinstance(new_diagrams, (str, bytes)):
            self.add_diagrams(pathlib.Path(new_diagrams))
        elif isinstance(new_diagrams, (tuple, list, set, frozenset, gen_type)):
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

        with self._lock, multiprocessing.pool.ThreadPool() as pool:
            idx, diags = zip(*((i, d) for i, d in self._diagrams.items() if isinstance(d, Diagram)))
            all_res = pool.starmap(func, tqdm.tqdm([(d, *args) for d in diags], unit="diag"))
            self._operations_chain.append((func, args))
        return dict(zip(idx, all_res))

    @property
    def diagrams(self) -> list[typing.Union[Diagram, None]]:
        """Return all the diagrams, pad missing by None."""
        with self._lock:
            return [d for d in self._diagrams.values() if isinstance(d, Diagram)]

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
                    self.add_diagram(diagram)
                    self._to_sniff["readed"].add(file)

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
