#!/usr/bin/env python3

"""Test the c rois optimisation."""

import gc
import tracemalloc

import numpy as np
import torch

from laueimproc.opti.rois import filter_by_indexs
from laueimproc.opti.rois import imgbboxes2raw


def test_memory_imgbboxes2raw():
    """Test imgbboxes2raw memory leak."""
    img = torch.rand((2000, 2000))
    bboxes = torch.zeros((1000, 4), dtype=torch.int32)
    bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40

    tracemalloc.start()
    for i in range(100):
        img_cpy, bboxes_cpy = img.clone(), bboxes.clone()
        imgbboxes2raw(img_cpy, bboxes_cpy).hex()  # copy to be shure countref >= 1
        (img_cpy, bboxes_cpy) = img_cpy.sum(), bboxes_cpy.sum()  # to be shure countref >= 1
        if i == 0:
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    top_stat = snapshot2.compare_to(snapshot1, "lineno")[0]
    print(top_stat.size_diff)
    assert top_stat.size_diff < 1_000, \
        f"imgbboxes2raw memory leak of {top_stat.size_diff} bytes"


def test_memory_filter_by_indexs():
    """Test imgbboxes2raw memory leak."""
    indexs = torch.tensor([1, 1, 0, -1, -2, *range(100, 1000)])
    bboxes = torch.zeros((1000, 4), dtype=torch.int32)
    bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40
    data = bytearray(np.linspace(0, 1, (bboxes[:, 2]*bboxes[:, 3]).sum(), dtype=np.float32).tobytes())

    tracemalloc.start()
    for i in range(100):
        indexs_cpy, data_cpy, bboxes_cpy = indexs.clone(), data.copy(), bboxes.clone()
        new_data, new_bboxes = filter_by_indexs(indexs_cpy, data_cpy, bboxes_cpy)
        new_dat = new_data.hex(), new_bboxes.sum()  # to be shure countref >= 1
        indexs_cpy, data_cpy, bboxes_cpy = indexs_cpy.sum(), data_cpy.hex(), bboxes_cpy.sum()  # to be shure countref >= 1
        if i == 0:
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    top_stat = snapshot2.compare_to(snapshot1, "lineno")[0]
    print(top_stat.size_diff)
    assert top_stat.size_diff < 1_000, \
        f"memory_filter_by_indexs memory leak of {top_stat.size_diff} bytes"

if __name__ == "__main__":
    test_memory_imgbboxes2raw()
    test_memory_filter_by_indexs()
