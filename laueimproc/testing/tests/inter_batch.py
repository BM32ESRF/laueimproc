#!/usr/bin/env python3

"""Test the median function."""

import pathlib
import tempfile

import cv2
import numpy as np
import pytest
import torch

from laueimproc.classes.dataset import DiagramsDataset


@pytest.mark.slow
def test_tol_random():
    """Create a random dataset, and compute the median."""
    folder = pathlib.Path(tempfile.mkdtemp())
    for i in range(101):
        img = torch.rand((2048, 2048))
        img *= 65335.0
        img += 0.5
        img_np = img.numpy().astype(np.uint16)
        assert cv2.imwrite(str(folder / f"rand_{i:04d}.tif"), img_np)

    dataset = DiagramsDataset(folder)
    median = dataset.compute_inter_image(level=0.5, method="sort")
    median_approx = dataset.compute_inter_image(level=0.5, method="snowflake", tol=1e-3)
    assert abs(median - median_approx).mean() < 1e-3
