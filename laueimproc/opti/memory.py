#!/usr/bin/env python3

"""Help to manage the memory."""


import pathlib
import psutil


def get_swappiness() -> int:
    """Return the system swapiness value."""
    file = pathlib.Path("/proc/sys/vm/swappiness")
    if not file.exists():
        return 60
    with open(file, "r", encoding="ascii") as raw:
        return int(raw.read())


def mem_to_free(swappiness: int = get_swappiness()) -> int:
    """Return the number of bytes to be removed from the cache."""
    assert isinstance(swappiness, int), swappiness.__class__.__name__
    assert 0 <= swappiness <= 100
    memory = psutil.virtual_memory()
    swappiness = max(5, min(95, swappiness))
    swap_limit = (100 - swappiness) * memory.total // 100
    size = max(0, (memory.total - swap_limit) - memory.available)
    return size
