#!/usr/bin/env python3

"""Help to manage the memory."""

import ctypes
import functools
import numbers
import os
import pathlib

import psutil


def free_malloc():
    """Clear the allocated malloc on linux."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

def get_swappiness() -> int:
    """Return the system swapiness value."""
    file = pathlib.Path("/proc/sys/vm/swappiness")
    if not file.exists():
        return 60
    with open(file, "r", encoding="ascii") as raw:
        return int(raw.read())

def mem_to_free(max_mem_percent: numbers.Integral) -> int:
    """Return the number of bytes to be removed from the cache.

    Parameters
    ----------
    max_mem_percent : int
        The maximum percent limit of memory to not excess. Value are in [0, 100].

    Returns
    -------
    mem_to_free : int
        The amount of memory to freed in order to reach the threshold
    """
    assert isinstance(max_mem_percent, numbers.Integral), max_mem_percent.__class__.__name__
    assert 0 <= max_mem_percent < 100, max_mem_percent
    max_mem_percent = max(5, min(95, max_mem_percent))
    threshold = total_memory() * max_mem_percent // 100
    size = max(0, used_memory() - threshold)
    return size

@functools.cache
def total_memory() -> int:
    """Return the total usable memory in bytes."""
    restricted_mem = os.environ.get("SLURM_MEM_PER_NODE", "0")
    restricted_mem = restricted_mem.replace("K", "e3")
    restricted_mem = restricted_mem.replace("M", "e6")
    restricted_mem = restricted_mem.replace("G", "e9")
    restricted_mem = restricted_mem.replace("T", "e12")
    restricted_mem = round(float(restricted_mem))
    restricted_mem = restricted_mem or psutil.virtual_memory().total
    return restricted_mem

def used_memory() -> int:
    """The total memory used in bytes."""
    if os.environ.get("SLURM_MEM_PER_NODE", "0") != "0":
        return psutil.Process().memory_info().rss
    memory = psutil.virtual_memory()
    return memory.total - memory.available
