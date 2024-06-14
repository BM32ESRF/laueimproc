#!/usr/bin/env python3

"""Compute the mean image of a batch of images."""

import multiprocessing.pool

from tqdm.autonotebook import tqdm
import torch

from laueimproc.classes.dataset import DiagramsDataset


def mean_stack(dataset: DiagramsDataset) -> torch.Tensor:
    """Compute the average image.

    Parameters
    ----------
    dataset : laueimproc.classes.dataset.DiagramsDataset
        The dataset containing all images.

    Returns
    -------
    torch.Tensor
        The average image of all the images contained in this dataset.
    """
    assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__
    assert len(dataset) > 0, "impossible to compute the mean image of an empty dataset"
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
            img += loc_img
    img /= float(len(indices))
    return img
