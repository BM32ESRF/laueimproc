#!/usr/bin/env python3

"""Read an image of laue diagram."""

import io
import logging
import lzma
import pathlib
import time
import warnings

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import torch
import yaml


def extract_metadata(data: bytes) -> dict:
    """Extract the metadata of the file content.

    Parameters
    ----------
    data : bytes
        The raw image file content.

    Returns
    -------
    metadata : dict
        The metadata of the file.
    """
    assert isinstance(data, bytes), data.__class__.__name__

    # extract
    metadata = {}
    if (
        data[4:8] in {b"jP  ", b"jP2 "}
        and (index := data.rfind(b"laueimproc_exif:")) != -1
    ):  # case jpeg2000, own safe and stable protocol
        try:
            metadata = lzma.decompress(data[index+16:])
        except lzma.LZMAError:
            logging.warning("failed to extract metadata of a jp2 file")
        else:
            metadata = yaml.safe_load(metadata)  # pickle is not security safe
            if not isinstance(metadata, dict):
                logging.warning("metadata of jp2 file is corrupted")
                metadata = {}
    else:
        try:
            img_pillow = PIL.Image.open(io.BytesIO(data))
        except PIL.UnidentifiedImageError:
            logging.warning("failed to extract metadata")
        else:
            metadata = {PIL.ExifTags.TAGS.get(k, k): v for k, v in img_pillow.getexif().items()}

    # filter
    for prop in (
        "ImageWidth",
        "ImageLength",
        "BitsPerSample",
        "Compression",
        "SamplesPerPixel",
        "TileWidth",
        "TileLength",
        "ColorMatrix1",
        "ColorMatrix2",
    ):
        if prop in metadata:
            del metadata[prop]

    return metadata


def read_image(filename: str | pathlib.Path) -> tuple[torch.Tensor, dict]:
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
    metadata : dict
        The file metadata.

    Raises
    ------
    OSError
        If the given path is not a file, or if the image reading failed.
    """
    assert isinstance(filename, str | pathlib.Path), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise OSError(f"the filename {filename} is not a file")

    # read image
    with open(filename, "rb") as stream:
        img_bytes = stream.read()
    buffer = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    if (image := cv2.imdecode(buffer, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)) is None:
        try:
            img_pillow = PIL.Image.open(io.BytesIO(img_bytes))
        except PIL.UnidentifiedImageError as err:
            raise OSError(f"failed to read {filename} with cv2 and pillow") from err
        image = np.asarray(img_pillow)
        if image.ndim != 2:
            image = np.asarray(PIL.ImageOps.grayscale(img_pillow))
    else:
        time.sleep(0.1)  # solve gil deadlock assumed to be durty
    image = to_floattensor(image)
    metadata = extract_metadata(img_bytes)

    return image, metadata


def to_floattensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
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
    return torch.asarray(data).clone()
