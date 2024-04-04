#!/usr/bin/env python3

"""Link serveral diagrams together."""

import inspect
import numbers
import pathlib
import pickle
import re
import threading
import typing
import warnings

from .diagram import Diagram


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
            if parameter.annotation is inspect._empty:
                warnings.warn("please specify the input type of `diag2ind`", SyntaxWarning)
            elif parameter.annotation is not Diagram:
                raise AssertionError(
                    f"the function `diag2ind` has to get a Diagram, not {parameter.annotation}"
                )
            if signature.return_annotation is inspect._empty:
                warnings.warn("please specify the return type of `diag2ind`", SyntaxWarning)
            elif signature.return_annotation is not int:
                raise AssertionError(
                    f"the function `diag2ind` has to return int, not {signature.return_annotation}"
                )
        self._diagrams: dict[Diagram] = {}  # index: diagram
        self._lock = threading.Lock()
        self.add_diagrams(diagram_refs)
        super().__init__(daemon=True)

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
                return self._diagrams[index]
            except KeyError as err:
                raise IndexError(f"The diagram of index {index} is not yet in the dataset") from err

    def __len__(self) -> int:
        """Return the numbers of diagrams present in the dataset."""
        return len(self._diagrams)

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
            return
        if isinstance(new_diagrams, pathlib.Path):
            assert new_diagrams.exists()
            raise NotImplementedError
        if hasattr(new_diagrams, "__iter__"):
            for new_diagram in new_diagrams:
                self.add_diagrams(new_diagram)
            return
        raise NotImplementedError

    def run(self):
        """Run asynchronousely in a child thread, called by self.start()."""
        raise NotImplementedError
