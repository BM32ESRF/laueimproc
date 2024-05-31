#!/usr/bin/env python3

"""Load and batch the data for neuronal network training."""


import logging
import numbers

import psutil
import torch

from laueimproc.classes.base_diagram import BaseDiagram
from laueimproc.opti.cache import auto_cache


NCPU = len(psutil.Process().cpu_affinity())


class SpotDataloader:
    """Get spots picture, apply dataaug and set in batch.

    Attributes
    ----------
    batch_size : int
        The batch dimension, read and write.
    """

    def __init__(self, dataset, model):
        """Initialise the data loader.

        Parameters
        ----------
        dataset : laueimproc.classes.dataset.DiagramsDataset
            Contains all the initialised diagrams.
        model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier
            The model, for the dataaug.
        """
        from laueimproc.classes.dataset import DiagramsDataset
        from laueimproc.nn.vae_spot_classifier import VAESpotClassifier
        assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__
        assert isinstance(model, VAESpotClassifier), model.__class__.__name__
        self.dataset = dataset
        self.model = model
        self._bath_size = None

        # find len
        self._diagram_indices = dataset.indices  # copy frozen
        self._nb_spots = sum(len(dataset[i]) for i in self._diagram_indices)
        if not self._nb_spots:
            raise ValueError(f"the dataset {dataset} does not contain any spots")

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the spot of this index."""
        i = 0
        for batch in self:
            if idx < i + len(batch):
                return batch[idx - i]
            i += len(batch)
        raise IndexError(f"no spot at index {idx}, len is {len(self)}")

    def __len__(self) -> int:
        """Return the total number of spots."""
        return self._nb_spots

    def __iter__(self) -> torch.Tensor:
        """Yield the batch of spots."""

        @auto_cache
        def batch_diag_spots(diagram: BaseDiagram) -> torch.Tensor:
            """Apply the dataaug on each spots and concatenate them."""
            if len(diagram) == 0:
                return torch.empty((0, *self.model.shape), dtype=torch.float32)
            return torch.cat([
                self.model.dataaug(roi[:height, :width]).unsqueeze(0)
                for roi, (height, width) in zip(diagram.rois, diagram.bboxes[:, 2:].tolist())
            ])

        batch = torch.empty((0, *self.model.shape), dtype=torch.float32)
        for idx in self._diagram_indices:
            sub_batch = batch_diag_spots(self.dataset[idx])
            batch = torch.cat([batch, sub_batch])
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if len(batch):
            yield batch

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        if self._bath_size is None:
            raise RuntimeError("please set `batch_size`")
        return self._bath_size

    @batch_size.setter
    def batch_size(self, new_batch_size: numbers.Integral):
        """Set the batch size."""
        assert isinstance(new_batch_size, numbers.Integral), new_batch_size.__class__.__name__
        assert new_batch_size >= 1, new_batch_size
        self._bath_size = int(new_batch_size)


def find_shape(dataset, percent: numbers.Real) -> tuple[int, int]:
    """Scan the shape of all the spots and deduce the best shape.

    Parameters
    ----------
    dataset : laueimproc.classes.dataset.DiagramsDataset
        Contains all the initialised diagrams.
    percent : float
        The percentage of spots smaller or equal to the shape returned.

    Returns
    -------
    height : int
        The height shape.
    width : int
        The width shape.

    Examples
    --------
    >>> import laueimproc
    >>> from laueimproc.nn.loader import find_shape
    >>> def init(diagram: laueimproc.Diagram):
    ...     diagram.find_spots()
    ...
    >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples())
    >>> _ = dataset.apply(init)
    >>> find_shape(dataset, 0.95)
    (15, 12)
    >>> find_shape(dataset, 0.5)
    (5, 5)
    >>>
    """
    from laueimproc.classes.dataset import DiagramsDataset  # avoid cyclic import
    assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__
    assert isinstance(percent, numbers.Real), percent.__class__.__name__
    assert 0 < percent <= 1, percent

    # compute histogram of shapes
    height_hist, width_hist = {}, {}
    for diagram in dataset:
        if not diagram.is_init():
            raise RuntimeError(
                "before calling the find_shape functional, initialize all the diagrams"
            )
        for (height, width) in diagram.bboxes[:, 2:].tolist():
            height_hist[height] = height_hist.get(height, 0) + 1
            width_hist[width] = width_hist.get(width, 0) + 1

    if not height_hist:
        logging.warning("no spots found, so it is impossible to find the best spot shape")
        height_hist, width_hist = {1: 1}, {1: 1}  # shape (1, 1)

    # compute cumsum function
    height_cumsum = torch.asarray(
        [height_hist.get(i, 0) for i in range(max(height_hist)+1)], dtype=torch.float32
    )
    width_cumsum = torch.asarray(
        [width_hist.get(i, 0) for i in range(max(width_hist)+1)], dtype=torch.float32
    )
    height_cumsum = torch.cumsum(height_cumsum, dim=0)
    width_cumsum = torch.cumsum(width_cumsum, dim=0)
    height_cumsum /= float(height_cumsum[-1])
    width_cumsum /= float(width_cumsum[-1])

    # inverse repartition function
    height = torch.argmin((height_cumsum <= percent).view(torch.uint8)).item()
    width = torch.argmin((width_cumsum <= percent).view(torch.uint8)).item()

    return height, width
