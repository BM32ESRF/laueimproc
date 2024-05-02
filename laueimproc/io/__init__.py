#!/usr/bin/env python3

"""Read an write the data on the harddisk, persistant layer."""

import functools
import pathlib

from laueimproc.io.download import get_samples


__all__ = ["get_sample", "get_samples"]


@functools.cache
def get_sample() -> pathlib.Path:
    """Return the path of the test image.

    Examples
    --------
    >>> from laueimproc.io import get_sample
    >>> get_sample().name
    'ge.jp2'
    >>> get_sample().exists()
    True
    >>>
    """
    return pathlib.Path(__file__).parent / "ge.jp2"
