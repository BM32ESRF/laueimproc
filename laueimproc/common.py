#!/usr/bin/env python3

"""Little common tools."""

import numbers


def bytes2human(size: numbers.Real) -> str:
    """Convert a size in bytes in readable human string."""
    assert isinstance(size, numbers.Real), size.__class__.__name__
    assert size >= 0, size
    unit, factor = {
        (True, True, True): ("GB", 1e-9),
        (False, True, True): ("MB", 1e-6),
        (False, False, True): ("kB", 1e-3),
        (False, False, False): ("B", 1.0),
    }[(size > 1e9, size > 1e6, size > 1e3)]
    return f"{size*factor:.1f}{unit}"
