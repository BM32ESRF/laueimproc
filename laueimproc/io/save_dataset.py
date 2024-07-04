#!/usr/bin/env python3

"""Write an load a dataset."""

import multiprocessing
import pathlib
import pickle


LOCK = multiprocessing.Lock()


def restore_dataset(filename: pathlib.Path, dataset=None):
    """Load or update the content of the dataset.

    Parameters
    ----------
    filename : pathlib.Path
        The filename of the pickle file, including the extension.
    dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset, optional
        If provided, update the state of the dataset by the recorded content.
        If it is not provided, a new dataset is created.

    Returns
    -------
    dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset
        A reference to the updated dataset.

    Examples
    --------
    >>> import pathlib
    >>> import tempfile
    >>> from laueimproc.io.save_dataset import restore_dataset, write_dataset
    >>> import laueimproc
    >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples())
    >>> dataset.state
    '047e5d6c00850898c233128e31e1f7e1'
    >>> file = pathlib.Path(tempfile.gettempdir()) / "dataset.pickle"
    >>> file = write_dataset(file, dataset)
    >>> empty_dataset = laueimproc.DiagramsDataset()
    >>> empty_dataset.state
    '746e927ad8595aa230adef17f990ba96'
    >>> filled_dataset = restore_dataset(file, empty_dataset)
    >>> filled_dataset.state
    '047e5d6c00850898c233128e31e1f7e1'
    >>> empty_dataset is filled_dataset  # update state inplace
    True
    >>>
    """
    from laueimproc.classes.base_dataset import BaseDiagramsDataset

    assert isinstance(filename, pathlib.Path), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".pickle")
    assert dataset is None or isinstance(dataset, BaseDiagramsDataset), dataset.__class__.__name__

    with LOCK, open(filename, "rb") as raw:
        new_dataset = pickle.load(raw)

    if dataset is None:
        dataset = new_dataset
    else:
        dataset.__setstate__(new_dataset.__getstate__())
    return dataset


def write_dataset(filename: str | pathlib.Path, dataset) -> pathlib.Path:
    """Write the pickle file associate to the provided diagrams dataset.

    Parameters
    ----------
    filename : pathlike
        The path to the file, relative or absolute.
        The suffix ".pickle" is automaticaly append if it is not already provided.
        If a file is already existing, it is overwriten.
    dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset
        The reference to the diagrams dataset.

    Returns
    -------
    filename : pathlib.Path
        The real absolute filename used.
    """
    from laueimproc.classes.base_dataset import BaseDiagramsDataset

    assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".pickle")
    assert isinstance(dataset, BaseDiagramsDataset), dataset.__class__.__name__

    with LOCK, open(filename, "wb") as raw:
        pickle.dump(dataset, raw)

    return filename
