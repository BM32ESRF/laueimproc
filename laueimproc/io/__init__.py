#!/usr/bin/env python3

"""Read an write the data on the harddisk, persistant layer."""

import functools
import pathlib

from laueimproc.io.download import get_samples


__all__ = ["get_samples", "get_test_sample"]


@functools.cache
def get_test_sample() -> pathlib.Path:
    """Return the path of the test image.

    Examples
    --------
    >>> from laueimproc.io import get_test_sample
    >>> get_test_sample().name
    'ge_blanc.jp2'
    >>> get_test_sample().exists()
    True
    >>>
    """
    return pathlib.Path(__file__).parent / "ge_blanc.jp2"
