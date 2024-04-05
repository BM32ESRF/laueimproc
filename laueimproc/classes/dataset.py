#!/usr/bin/env python3

"""Link serveral diagrams together."""

import inspect
import multiprocessing.pool
import numbers
import pathlib
import pickle
import re
import threading
import time
import traceback
import typing
import warnings

import tqdm

from .diagram import Diagram


def _excepthook(args):
    """Raise the exceptions comming from a DiagramDataset thread."""
    if isinstance(args.thread, DiagramDataset):
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


class DiagramDataset(threading.Thread):
    """Link Diagrams together.

    Attributes
    ----------
    diagrams : list[laueimproc.classes.diagram.Diagram]
        All the diagrams contained in the dataset.
    """

    def __init__(
        self, *diagram_refs, diag2ind: typing.Optional[typing.Callable[[Diagram], int]] = None
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
        """
        if diag2ind is None:
            self._diag2ind = default_diag2ind
        else:
            assert callable(diag2ind), \
                f"`diag2ind` has to be callable, not {diag2ind.__class__.__name__}"
            try:
                pickle.dumps(diag2ind)
            except pickle.PicklingError as err:
                raise AssertionError(
                    f"the function `diag2ind` {diag2ind} is not pickalable"
                ) from err
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
            elif signature.return_annotation is not int:
                raise AssertionError(
                    f"the function `diag2ind` has to return int, not {signature.return_annotation}"
                )
        self._diagrams: dict[int, Diagram] = {}  # index: diagram
        self._lock = threading.Lock()
        self._to_sniff: dict = {"dirs": [], "readed": set()}  # for the async run method
        self._operations_chain: list[typing.Callable[[Diagram], object]] = []
        self.add_diagrams(diagram_refs)
        super().__init__(daemon=True)
        self.start()

    def __getitem__(self, index: numbers.Integral) -> Diagram:
        """Get the diagram of index `index`.

        Parameters
        ----------
        index : int
            The index of the diagram you want to reach.

        Raises
        ------
        IndexError
            If the diagram of index `index` is not founded yet.
        """
        assert isinstance(index, numbers.Integral), index.__class__.__name__
        assert index >= 0, f"only positive indexs are allowed, not {index}"
        index = int(index)
        with self._lock:
            try:
                diagram = self._diagrams[index]
            except KeyError as err:
                raise IndexError(f"The diagram of index {index} is not yet in the dataset") from err
        if not isinstance(diagram, Diagram):
            raise IndexError(f"The diagram of index {index} is comming soon in the dataset")
        return diagram

    def __len__(self) -> int:
        """Return the numbers of diagrams present in the dataset."""
        return len(self.diagrams)

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
        index = self._diag2ind(new_diagram)
        assert isinstance(index, numbers.Integral), (
            f"the function {self._diag2ind} must return an integer, "
            f"not a {index.__class__.__name__}"
        )
        assert index >= 0, \
            f"the function {self._diag2ind} must return a positive number, not {index}"
        index = int(index)
        with self._lock:
            if index in self._diagrams:
                raise LookupError(f"the diagram of index {index} is already present in the dataset")
            for func, args in self._operations_chain:  # TODO make it async
                func(new_diagram, *args)
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
        if isinstance(new_diagrams, Diagram):
            self.add_diagram(new_diagrams)
        elif isinstance(new_diagrams, pathlib.Path):
            new_diagrams.expanduser().resolve()
            assert new_diagrams.exists(), f"{new_diagrams} if not an existing path"
            if new_diagrams.is_dir():
                self._to_sniff["dirs"].append([new_diagrams, 0.0])
            else:
                self.add_diagram(Diagram(new_diagrams))  # case file
        elif isinstance(new_diagrams, (str, bytes)):
            self.add_diagrams(pathlib.Path(new_diagrams))
        elif isinstance(new_diagrams, (tuple, list, set, frozenset)):
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
    def diagrams(self) -> list[Diagram]:
        """Return all the diagrams contained in the dataset."""
        with self._lock:
            return [d for d in self._diagrams.values() if d is not None]

    def run(self):
        """Run asynchronousely in a child thread, called by self.start()."""
        while True:
            # add new diagrams
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
            time.sleep(10)
