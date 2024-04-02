#!/usr/bin/env python3


"""Test the c rois optimisation."""

import gc
import tracemalloc

import torch

from laueimproc.opti.rois import imgbboxes2raw


def test_memory_imgbboxes2raw():
    """Test imgbboxes2raw memory leak."""
    img = torch.rand((2000, 2000))
    bboxes = torch.zeros((1000, 4), dtype=torch.int32)
    bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40

    tracemalloc.start()
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    for _ in range(10):
        img_copy, bboxes_copy = img.clone(), bboxes.clone()
        imgbboxes2raw(img_copy, bboxes_copy).copy()  # copy to be shure countref >= 1
        img_copy.clone(), bboxes_copy.clone()  # to be shure countref >= 1
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    top_stat = snapshot2.compare_to(snapshot1, "lineno")[0]
    # print(top_stat.size_diff)
    assert top_stat.size_diff < 1_000, \
        f"imgbboxes2raw memory leak of {top_stat.size_diff} bytes"


if __name__ == "__main__":
    test_memory_imgbboxes2raw()
