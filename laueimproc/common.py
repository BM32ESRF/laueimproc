#!/usr/bin/env python3

"""Little common tools."""

import functools
import numbers
import pathlib
import re
import typing


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


def time2sec(time: typing.Union[numbers.Real, str], /) -> float:
    """Parse a time duration expression and return it in seconds.

    Raises
    ------
    ValueError
        If the provided time dosen't match a parsable correct time format.

    Examples
    --------
    >>> from laueimproc.common import time2sec
    >>> time2sec(12.34)
    12.34
    >>> time2sec("12.34")
    12.34
    >>> time2sec(".34")
    0.34
    >>> time2sec("12.")
    12.0
    >>> time2sec("12")
    12.0
    >>> time2sec("12.34s")
    12.34
    >>> time2sec("12.34 sec")
    12.34
    >>> time2sec("2 m")
    120.0
    >>> time2sec("2min 2")
    122.0
    >>> time2sec("  2.5  h  ")
    9000.0
    >>> time2sec("2hour02")
    7320.0
    >>> time2sec("2h 2s")
    7202.0
    >>> time2sec("2.5 hours 2.0 minutes 12.34 seconds")
    9132.34
    >>>
    """
    if isinstance(time, numbers.Real):
        if time < 0:
            raise ValueError(f"the time has to be positive, noe {time}")
        return float(time)
    assert isinstance(time, str), time.__class__.__name__

    pat_float = r"(?:\d++\.?+\d*+)|(?:\.\d++)"
    if (match := re.match(rf"^\s*(?P<seconds>{pat_float})\s*$", time)) is not None:
        return float(match["seconds"])
    pat_hms = re.compile(
        (
            r"^\s*"
            rf"(?:(?P<hours>{pat_float})\s*(?:hours|hour|h))?\s*"
            rf"(?:(?P<minutes>{pat_float})\s*(?:minutes|minute|min|m)?)?\s*"
            rf"(?:(?P<seconds>{pat_float})\s*(?:seconds|second|sec|s)?)?"
            r"\s*$"
        ),
        re.IGNORECASE,
    )
    if (match := re.match(pat_hms, time)) is not None:
        hours = float(match["hours"] or "0")
        minutes = float(match["minutes"] or "0")
        seconds = float(match["seconds"] or "0")
        return 3600*hours + 60*minutes + seconds
    raise ValueError(f"failed to parse the time duration {repr(time)}")
