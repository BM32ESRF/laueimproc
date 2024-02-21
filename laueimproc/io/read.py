#!/usr/bin/env python3

"""Read an image of laue diagram."""

import pathlib
import typing

import cv2
import fabio
import numpy as np


def read_image(filename: typing.Union[str, bytes, pathlib.Path]) -> np.ndarray:
    """Read and decode a grayscale image into a numpy array.

    Use cv2 as possible, and fabio if cv2 failed.

    Parameters
    ----------
    filename : pathlike
        The path to the image, relative or absolute.

    Returns
    -------
    np.ndarray
        The grayscale laue image matrix in uint16.

    Raises
    ------
    OSError
        If the given path is not a file, or if the image reading failed.
    """
    assert isinstance(filename, (str, bytes, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise OSError(f"the filename {filename} is not a file")

    if (image := cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)) is not None:
        return image # cv2 9% faster than fabio on .mccd files
    try:
        with fabio.open(filename) as raw:
            image = raw.data
    except KeyError as err:
        raise OSError(f"failed to read {filename} with cv2 and fabio") from err
    return image
