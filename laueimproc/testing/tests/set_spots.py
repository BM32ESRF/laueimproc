#!/usr/bin/env python3

"""Test the set_spots function."""

import itertools
import random

import numpy as np
import pytest
import torch

from laueimproc.classes.base_diagram import BaseDiagram
from laueimproc.io import get_sample


def _validate_bboxes(diag: BaseDiagram, bboxes: torch.tensor):
    """Test if the bboxes are the same as the ref."""
    assert diag.is_init()
    assert torch.equal(diag.bboxes, bboxes)


def test_from_anchors_rois_numpy():
    """Init a diagram from anchors and rois."""
    diag = BaseDiagram(get_sample())
    anchors = itertools.product(range(15, min(diag.image.shape)-30, 200), repeat=2)
    rois = [np.empty((random.randint(5, 30), random.randint(5, 30)), dtype=np.uint16)]
    anchors_rois = [[i, j, roi] for (i, j), roi in zip(anchors, rois)]
    diag.set_spots(anchors_rois)
    assert len(diag) == len(anchors_rois)


def test_from_anchors_rois_torch():
    """Init a diagram from anchors and rois."""
    diag = BaseDiagram(get_sample())
    anchors = itertools.product(range(15, min(diag.image.shape)-30, 200), repeat=2)
    rois = [torch.zeros((random.randint(5, 30), random.randint(5, 30)))]
    anchors_rois = [[i, j, roi] for (i, j), roi in zip(anchors, rois)]
    diag.set_spots(anchors_rois)
    assert len(diag) == len(anchors_rois)


def test_from_bboxes_list():
    """Init a diagram from bboxes."""
    bboxes = [[0, 0, 10, 15], [100, 200, 25, 20]]

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))

    diag = BaseDiagram(get_sample())
    diag.set_spots(*bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots([0.0, 0, 10, 15])
    with pytest.raises(ValueError):
        diag.set_spots([0, 0, 10.0, 15])


def test_from_bboxes_numpy():
    """Init a diagram from bboxes."""
    bboxes = np.array([(0, 0, 10, 15), (100, 200, 25, 20)])

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots((0.0, 0, 10, 15))
    with pytest.raises(ValueError):
        diag.set_spots((0, 0, 10.0, 15))


def test_from_bboxes_set():
    """Init a diagram from bboxes."""
    bboxes = {(0, 0, 10, 15), (100, 200, 25, 20)}

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))

    diag = BaseDiagram(get_sample())
    diag.set_spots(frozenset(bboxes))
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))


def test_from_bboxes_torch():
    """Init a diagram from bboxes."""
    bboxes = torch.tensor([(0, 0, 10, 15), (100, 200, 25, 20)])

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, bboxes.to(torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots(bboxes.to(float))
    with pytest.raises(ValueError):
        diag.set_spots(bboxes.to(complex))


def test_from_bboxes_tuple():
    """Init a diagram from bboxes."""
    bboxes = [(0, 0, 10, 15), (100, 200, 25, 20)]

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))

    diag = BaseDiagram(get_sample())
    diag.set_spots(*bboxes)
    _validate_bboxes(diag, torch.astype(bboxes, torch.int16))


def test_from_diagram():
    """Init a diagram from an other."""
    # preparation
    diag_ref = BaseDiagram(get_sample())
    diag_ref.find_spots()
    # tests
    diag = BaseDiagram(get_sample())
    diag.set_spots(diag_ref)
    assert torch.equal(diag_ref.rois, diag.rois)


def test_reset():
    """Init diagram with empty data."""
    diag = BaseDiagram(get_sample())
    diag.set_spots()
    assert diag.is_init()
    assert len(diag) == 0

    diag = BaseDiagram(get_sample())
    diag.set_spots([])
    assert diag.is_init()
    assert len(diag) == 0

    diag = BaseDiagram(get_sample())
    diag.set_spots([[]])
    assert diag.is_init()
    assert len(diag) == 0
