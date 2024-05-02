#!/usr/bin/env python3

"""Read an image of laue diagram."""

import pathlib
import typing
import warnings

import cv2
import fabio
import numpy as np
import torch


def read_image(filename: typing.Union[str, pathlib.Path]) -> torch.Tensor:
    """Read and decode a grayscale image into a numpy array.

    Use cv2 as possible, and fabio if cv2 failed.

    Parameters
    ----------
    filename : pathlike
        The path to the image, relative or absolute.

    Returns
    -------
    image : torch.Tensor
        The grayscale laue image matrix in float between with value range in [0, 1].

    Raises
    ------
    OSError
        If the given path is not a file, or if the image reading failed.
    """
    assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise OSError(f"the filename {filename} is not a file")

    if (image := cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)) is not None:
        return to_floattensor(image)  # cv2 9% faster than fabio on .mccd files
    try:
        with fabio.open(filename) as raw:
            image = raw.data
    except KeyError as err:
        raise OSError(f"failed to read {filename} with cv2 and fabio") from err
    return to_floattensor(image)


def to_floattensor(data: typing.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert and shift tenso into float torch tensor.

    If the input is not in floating point, it is converting in float32
    and the value range is set beetweeen 0 and 1.

    Parameters
    ----------
    data : arraylike
        The torch tensor or the numpy array to convert.

    Returns
    -------
    tensor : torch.Tensor
        The float torch tensor.
    """
    if isinstance(data, torch.Tensor):
        if not data.dtype.is_floating_point:
            iinfo = torch.iinfo(data.dtype)
            data = data.to(dtype=torch.float32)
            data -= float(iinfo.min)
            data *= 1.0 / float(iinfo.max - iinfo.min)
        return data

    # we have to convert in float from numpy (no torch delegation)
    # because some dtype are not supported by torch
    if isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.floating):
            iinfo = np.iinfo(data.dtype)
            data = data.astype(np.float32)
            data -= float(iinfo.min)
            data *= 1.0 / float(iinfo.max - iinfo.min)
        return torch.from_numpy(data)  # no copy

    warnings.warn(
        "to instanciate a image from a non arraylike data will be forbiden", DeprecationWarning
    )
    return torch.tensor(data)  # copy
