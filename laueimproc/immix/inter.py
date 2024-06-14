#!/usr/bin/env python3

"""Compute an intermediate image of a batch of images."""

import math
import multiprocessing.pool
import numbers
import threading
import typing

from tqdm.autonotebook import tqdm
import torch

from laueimproc.classes.dataset import DiagramsDataset


class MomentumMixer:
    """Compute an average momenttume betweem the two closest candidates.

    Attributes
    ----------
    high : int
        The index max of the sorted stack.
    low : int
        The index min of the sorted stack.
    """

    def __init__(self, nbr: int, level: float):
        """Precompute the momentum.

        Parameters
        ----------
        nbr : int
            The number of items in the stack.
        level : float
            The relative position in the stack in [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.immix.inter import MomentumMixer
        >>> mom = MomentumMixer(5, 0.5)
        >>> mom(torch.arange(5)[mom.low], torch.arange(5)[mom.high])
        tensor(2.)
        >>> mom = MomentumMixer(4, 0.5)
        >>> mom(torch.arange(4)[mom.low], torch.arange(4)[mom.high])
        tensor(1.5000)
        >>> mom = MomentumMixer(101, 0.1*torch.pi)
        >>> mom(torch.arange(101)[mom.low], torch.arange(101)[mom.high])
        tensor(31.4159)
        >>>
        """
        assert isinstance(nbr, int), nbr.__class__.__name__
        assert nbr >= 1, nbr
        assert isinstance(level, float), level.__class__.__name__
        assert 0.0 <= level <= 1.0, level

        if nbr == 1:
            self._low = self._high = 0
            self._momentum = 0.0

        interval = 1.0 / (nbr - 1)
        abs_level = level / interval
        self._low = math.floor(abs_level)
        self._low = max(0, min(nbr - 2, self._low))
        self._high = self._low + 1
        self._momentum = abs_level - self._low

    def __call__(self, val_min, val_max):
        """Combine the two elements."""
        comb = val_min * (1.0 - self._momentum)
        comb += val_max * self._momentum
        return comb

    @property
    def high(self) -> int:
        """Return the index max of the sorted stack."""
        return self._high

    @property
    def low(self) -> int:
        """Return the index min of the sorted stack."""
        return self._low


def _best_idtype(nbr):
    if nbr <= torch.iinfo(torch.int16).max:
        return torch.int16
    if nbr <= torch.iinfo(torch.int32).max:
        return torch.int32
    return torch.int64


def _extrema_stack(dataset: DiagramsDataset, func: typing.Callable) -> torch.Tensor:
    """Find the min filter of a stack of images."""
    indices = dataset.indices
    img = None
    with multiprocessing.pool.ThreadPool() as pool:
        for loc_img in tqdm(
            pool.imap_unordered(lambda i: dataset[i].image, indices),
            total=len(indices),
            unit="img",
            desc="mean image",
        ):
            if img is None:
                img = loc_img.clone()
            else:
                assert img.shape == loc_img.shape, \
                    f"all images have be of same shape ({img.shape} vs {loc_img.shape})"
            img = func(img, loc_img, out=img)
    return img


def _snowflake_stack(diagram, hist, b_min, res, lock):
    image = diagram.image.ravel()
    idx = 1 + ((image - b_min) / res).to(torch.int64)  # floor
    idx = torch.clamp(idx, 0, hist.shape[1] - 1, out=idx)
    with lock:
        hist[hist.range, idx] += 1


def snowflake_stack(
    dataset: DiagramsDataset, level: numbers.Real = 0.5, tol: numbers.Real = 1.526e-5
) -> torch.Tensor:
    """Compute the median, first quartile, third quartile or everything in between.

    This algorithm consists of computing the histogram of all the images into a heap of size n.
    Then compute the cumulative histogram to deduce in each slice the value is. To bound the result.
    Iterate the processus to refine the bounds until reaching the required accuracy.

    Parameters
    ----------
    dataset : laueimproc.classes.dataset.DiagramsDataset
        The dataset containing all images.
    level : float, default=0.5
        The level of the sorted stack.

        * 0 -> min filter
        * 0.25 -> first quartile
        * 0.5 (default) -> median
        * 0.75 -> third quartile
        * 1 -> max filter
    tol : float, default=1/(2**16-1)
        Accuracy of the estimated returned image.

    Returns
    -------
    torch.Tensor
        The 2d float32 grayscale image.

    Notes
    -----
    Unlike the native algorithm, images are read a number of times proportional
    to the logarithm of the inverse of the precision.
    Independent of the number of images in the dataset.
    This algorithm is therefore better suited to large datasets.
    """
    assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__
    assert len(dataset) > 0, "impossible to compute the inter image of an empty dataset"
    assert isinstance(level, numbers.Real), level.__class__.__name__
    assert 0 <= level <= 1, level

    if float(level) in {0.0, 1.0}:
        return _extrema_stack(dataset, {0.0: torch.minimum, 1.0: torch.maximum}[float(level)])

    # initialization
    indices = dataset.indices
    height, width = dataset[indices[0]].image.shape
    hist = torch.empty(
        height * width,
        max(4, 100_000_000 // (_best_idtype(len(indices)).itemsize * height * width)),  # 100 Mo
        dtype=_best_idtype(len(indices)),
    )
    hist.range = torch.arange(len(hist), dtype=torch.int64, device=hist.device)
    b_min, b_max = torch.full((height * width,), 0.0), torch.full((height * width,), 1.0)
    res = 1.0 / (hist.shape[1] - 2)
    mux = MomentumMixer(len(indices), float(level))
    lock = threading.Lock()

    # main
    for _ in tqdm(
        range(max(0, math.ceil(-math.log(tol) / math.log(hist.shape[1] - 2)) - 1)),
        desc="inter image", unit="loop",
    ):
        hist[...] = 0
        res = (b_max - b_min) / (hist.shape[1] - 2)
        with multiprocessing.pool.ThreadPool() as pool:
            for _ in tqdm(
                pool.starmap(
                    _snowflake_stack,
                    ((dataset[i], hist, b_min, res, lock) for i in indices),
                ),
                total=len(indices),
                unit="img",
                leave=False,
                position=1,
            ):
                pass
        hist = torch.cumsum(hist, dim=1, out=hist)
        b_min_int = torch.argmax((hist >= mux.low + 1).view(torch.uint8), dim=1)
        # b_max_int = torch.argmax((hist >= mux.high + 1).view(torch.uint8), dim=1)
        b_max_int = b_min_int
        b_max = b_max_int.to(res.dtype) * res + b_min
        b_min += (b_min_int - 1).to(res.dtype) * res

    res *= 0.5
    return mux(b_min + res, b_max - res).reshape(height, width)


def sort_stack(dataset: DiagramsDataset, level: numbers.Real = 0.5) -> torch.Tensor:
    """Compute the median, first quartile, third quartile or everything in between.

    This algorithm consists of stacking all the images into a heap of size n.
    Then sort each column in the stack (as many columns as there are pixels in the image).
    Finally, we return the image in the new stack at height n * `level`.

    Parameters
    ----------
    dataset : laueimproc.classes.dataset.DiagramsDataset
        The dataset containing all images.
    level : float, default=0.5
        The level of the sorted stack.

        * 0 -> min filter
        * 0.25 -> first quartile
        * 0.5 (default) -> median
        * 0.75 -> third quartile
        * 1 -> max filter

    Returns
    -------
    torch.Tensor
        The 2d float32 grayscale image.

    Notes
    -----
    For reasons of memory limitations, the final image is calculated in small chunks.
    As a result, each image on the hard disk is read n times,
    with n proportional to the number of diagrams in the dataset.
    """
    assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__
    assert len(dataset) > 0, "impossible to compute the inter image of an empty dataset"
    assert isinstance(level, numbers.Real), level.__class__.__name__
    assert 0 <= level <= 1, level

    if float(level) in {0.0, 1.0}:
        return _extrema_stack(dataset, {0.0: torch.minimum, 1.0: torch.maximum}[float(level)])

    # initialization
    indices = dataset.indices
    height, width = dataset[indices[0]].image.shape
    buff_size = max(1, 100_000_000 // (torch.float32.itemsize * len(dataset)))  # 100 Mo
    out = torch.empty(height * width, dtype=torch.float32)
    mux = MomentumMixer(len(indices), float(level))

    def _select(diagram_start_end):
        diagram, start, end = diagram_start_end
        # .clone() allows gc to collect source image
        return diagram.image.ravel()[None, start:end].clone()

    # main
    for start in tqdm(range(0, height * width, buff_size), desc="inter image", unit="batch"):
        end = min(height * width, start + buff_size)
        with multiprocessing.pool.ThreadPool() as pool:
            batch = torch.cat(
                list(
                    tqdm(
                        pool.imap_unordered(
                            _select, ((dataset[i], start, end) for i in indices)
                        ),
                        total=len(indices),
                        unit="img",
                        leave=False,
                        position=1,
                    )
                )
            )
        batch = torch.sort(batch, dim=0).values
        out[start:end] = mux(batch[mux.low, :], batch[mux.high, :])
    return out.reshape(height, width)
