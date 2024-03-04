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


def mem_to_free(ram_limit: float) -> int:
    """Return the number of bytes to be removed from the cache."""
    assert isinstance(ram_limit, float), ram_limit.__class__.__name__
    assert 0 < ram_limit < 1
    memory = psutil.virtual_memory()
    ram_limit = max(0.05, min(0.95, ram_limit))
    swap_limit = round(ram_limit * memory.total)
    size = max(0, (memory.total - swap_limit) - memory.available)
    return size
