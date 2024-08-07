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
    anchors_rois = [
        [i, j, np.empty((random.randint(5, 30), random.randint(5, 30)), dtype=np.uint16)]
        for i, j in anchors
    ]
    diag.set_spots(anchors_rois)
    assert len(diag) == len(anchors_rois)


def test_from_anchors_rois_torch():
    """Init a diagram from anchors and rois."""
    diag = BaseDiagram(get_sample())
    anchors = itertools.product(range(15, min(diag.image.shape)-30, 200), repeat=2)
    anchors_rois = [
        [i, j, np.empty((random.randint(5, 30), random.randint(5, 30)), dtype=np.uint16)]
        for i, j in anchors
    ]
    diag.set_spots(anchors_rois)
    assert len(diag) == len(anchors_rois)


def test_from_bboxes_list():
    """Init a diagram from bboxes."""
    bboxes = [[0, 0, 10, 15], [100, 200, 25, 20]]

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.asarray(bboxes).to(torch.int16))

    diag = BaseDiagram(get_sample())
    diag.set_spots(*bboxes)
    _validate_bboxes(diag, torch.asarray(bboxes).to(torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots([0.0, 0, 10, 15])
    with pytest.raises(ValueError):
        diag.set_spots([0, 0, 10.0, 15])


def test_from_bboxes_numpy():
    """Init a diagram from bboxes."""
    bboxes = np.array([(0, 0, 10, 15), (100, 200, 25, 20)])

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.from_numpy(bboxes).to(torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots((0.0, 0, 10, 15))
    with pytest.raises(ValueError):
        diag.set_spots((0, 0, 10.0, 15))


def test_from_bboxes_set():
    """Init a diagram from bboxes."""
    bboxes = {(0, 0, 10, 15), (100, 200, 25, 20)}

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    assert len(diag) == 2  # order not guaranty

    diag = BaseDiagram(get_sample())
    diag.set_spots(frozenset(bboxes))
    assert len(diag) == 2  # order not guaranty


def test_from_bboxes_torch():
    """Init a diagram from bboxes."""
    bboxes = torch.tensor([(0, 0, 10, 15), (100, 200, 25, 20)])

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, bboxes.to(torch.int16))

    with pytest.raises(ValueError):
        diag.set_spots(bboxes.to(torch.float32))
    with pytest.raises(ValueError):
        diag.set_spots(bboxes.to(torch.complex64))


def test_from_bboxes_tuple():
    """Init a diagram from bboxes."""
    bboxes = [(0, 0, 10, 15), (100, 200, 25, 20)]

    diag = BaseDiagram(get_sample())
    diag.set_spots(bboxes)
    _validate_bboxes(diag, torch.asarray(bboxes).to(torch.int16))

    diag = BaseDiagram(get_sample())
    diag.set_spots(*bboxes)
    _validate_bboxes(diag, torch.asarray(bboxes).to(torch.int16))


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
