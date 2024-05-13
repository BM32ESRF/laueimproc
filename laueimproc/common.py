#!/usr/bin/env python3

"""Little common tools."""

import functools
import numbers
import pathlib


def bytes2human(size: numbers.Real) -> str:
    """Convert a size in bytes in readable human string.

    Examples
    --------
    >>> from laueimproc.common import bytes2human
    >>> bytes2human(0)
    '0.0B'
    >>> bytes2human(2000)
    '2.0kB'
    >>> bytes2human(2_000_000)
    '2.0MB'
    >>> bytes2human(2e9)
    '2.0GB'
    >>>
    """
    assert isinstance(size, numbers.Real), size.__class__.__name__
    assert size >= 0, size
    unit, factor = {
        (True, True, True): ("GB", 1e-9),
        (False, True, True): ("MB", 1e-6),
        (False, False, True): ("kB", 1e-3),
        (False, False, False): ("B", 1.0),
    }[(size > 1e9, size > 1e6, size > 1e3)]
    return f"{size*factor:.1f}{unit}"


@functools.cache
def get_project_root() -> pathlib.Path:
    """Return the absolute project root folder.

    Examples
    --------
    >>> from laueimproc.common import get_project_root
    >>> root = get_project_root()
    >>> root.is_dir()
    True
    >>> root.name
    'laueimproc'
    >>>
    """
    return pathlib.Path(__file__).resolve().parent
