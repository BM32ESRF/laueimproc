#!/usr/bin/env python3

"""Provide tools for switching convention.

The two conventions supported are:

* `ij`: Extension by continuity (N -> R) of the numpy convention (height, width).
The first axis iterates on lines from top to bottom, the second on columns from left to right.
In an image, the origin (i=0, j=0) correspond to the top left image corner of the top left pixel.
It means that the center of the top left pixel has the coordinate (i=1/2, j=1/2).

* `xy`: A transposition and a translation of the origin of the `ij` convention.
The first axis iterates on columns from left to right, the second on lines from top to bottom.
In an image, the point (x=1, y=1) correspond to the middle of the top left pixel.
"""

import functools
import typing

import numpy as np
import torch


def ij_to_xy(
    array: typing.Union[torch.Tensor, np.ndarray],
    *,
    i: typing.Union[tuple[int, slice, type(Ellipsis)], int, slice, type(Ellipsis)],
    j: typing.Union[tuple[int, slice, type(Ellipsis)], int, slice, type(Ellipsis)],
) -> typing.Union[torch.Tensor, np.ndarray]:
    """Switch the axis i and j, and append 1/2 to all values.

    Parameters
    ----------
    array : torch.Tensor or np.ndarray
        The data in ij convention.
    i, j : tuple, int, slice or Ellipsis
        The indexing of the i subdata and j subdata.

    Returns
    -------
    array : torch.Tensor or np.ndarray
        A reference to the ij_array, with the axis converted in xy convention.

    Notes
    -----
    Input and output data are shared in place.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.convention import ij_to_xy
    >>> array = torch.zeros((10, 2))
    >>> array[:, 0] = torch.linspace(0, 1, 10)  # i axis
    >>> array[:, 1] = torch.linspace(2, 1, 10)  # j axis
    >>> array
    tensor([[0.0000, 2.0000],
            [0.1111, 1.8889],
            [0.2222, 1.7778],
            [0.3333, 1.6667],
            [0.4444, 1.5556],
            [0.5556, 1.4444],
            [0.6667, 1.3333],
            [0.7778, 1.2222],
            [0.8889, 1.1111],
            [1.0000, 1.0000]])
    >>> ij_to_xy(array, i=(..., 0), j=(..., 1))
    tensor([[2.5000, 0.5000],
            [2.3889, 0.6111],
            [2.2778, 0.7222],
            [2.1667, 0.8333],
            [2.0556, 0.9444],
            [1.9444, 1.0556],
            [1.8333, 1.1667],
            [1.7222, 1.2778],
            [1.6111, 1.3889],
            [1.5000, 1.5000]])
    >>> _ is array  # inplace
    True
    >>>
    """
    assert isinstance(array, (torch.Tensor, np.ndarray)), array.__class__.__name__
    ydata = array[i] + 0.5  # copy
    xdata = array[j]  # reference
    xdata += 0.5  # reference
    array[i] = xdata  # copy
    array[j] = ydata  # copy
    return array


def ij_to_xy_decorator(
    i: typing.Union[tuple[int, slice, type(Ellipsis)], int, slice, type(Ellipsis)],
    j: typing.Union[tuple[int, slice, type(Ellipsis)], int, slice, type(Ellipsis)],
):
    """Append the argument conv to a function to allow user switching convention."""
    def decorator(func: callable):
        @functools.wraps(func)
        def decorated(*args, conv: str = "ij", **kwargs):
            assert isinstance(conv, str), conv.__class__.__name__
            assert conv in {"ij", "xy"}, conv
            array = func(*args, **kwargs)  # assumed to be in ij convention
            if conv != "ij":
                array = globals()[f"ij_to_{conv}"](array, i=i, j=j)
            return array
        return decorated
    return decorator
