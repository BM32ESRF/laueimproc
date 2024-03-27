#!/usr/bin/env python3

"""Compress the rois for storage."""

import lzma
import struct

import numpy as np
import torch


def compress_rois(rois: torch.Tensor) -> bytes:
    """Reduce the size of the tensor.

    Parameters
    ----------
    rois : Tensor
        A float32 images of shape (n, h, w) with value in range [0, 1].
    """
    # compression
    rois_16b = (rois.numpy(force=True)*65535.0 + 0.5).astype(np.uint16)  # +0.5 for round, not floor
    # convert into binary
    brut_data = struct.pack("LHH", *rois.shape) + rois_16b.tobytes()
    # compressed (no zlib because 90 ms vs 40 ms but comp ratio 110 vs 85)
    comp_data = lzma.compress(brut_data, preset=5, format=lzma.FORMAT_XZ)
    # comp_data = zlib.compress(brut_data, level=9, wbits=31)
    return comp_data

def decompress_rois(comp_data: bytes) -> torch.Tensor:
    """Inverse operation of ``compress_rois``."""
    brut_data = lzma.decompress(comp_data, format=lzma.FORMAT_XZ)
    # brut_data = zlib.decompress(comp_data, wbits=31)
    header = struct.calcsize("LHH")
    shape = struct.unpack("LHH", brut_data[:header])
    rois_16b = np.frombuffer(brut_data[header:], dtype=np.uint16).reshape(shape)
    rois = torch.from_numpy(rois_16b.astype(np.float32))
    rois /= 65535.0
    return rois
