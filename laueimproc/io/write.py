#!/usr/bin/env python3

"""Write an image of laue diagram."""

import lzma
import pathlib
import typing

import cv2
import numpy as np
import torch
import yaml


def write_jp2(
    filename: typing.Union[str, pathlib.Path],
    image: torch.Tensor,
    metadata: typing.Optional[dict] = None,
):
    """Write a lossless jpeg2000 image with the metadata.

    Parameters
    ----------
    filename : pathlike
        The path to the file, relative or absolute.
        The suffix ".jp2" is automaticaly append if it is not already provided.
        If a file is already existing, it is overwriten.
    image : torch.Tensor
        The 2d grayscale float image with values in range [0, 1].
    metadata : dict, optional
        If provided, a jsonisable dictionary of informations.

    Examples
    --------
    >>> import pathlib
    >>> import tempfile
    >>> import torch
    >>> from laueimproc.io.read import read_image
    >>> from laueimproc.io.write import write_jp2
    >>>
    >>> file = pathlib.Path(tempfile.gettempdir()) / "image.jp2"
    >>> img_ref = torch.rand((2048, 1024))
    >>> img_ref = (img_ref * 65535 + 0.5).to(int).to(torch.float32) / 65535
    >>> metadata_ref = {"prop1": 1}
    >>>
    >>> write_jp2(file, img_ref, metadata_ref)
    >>> img, metadata = read_image(file)
    >>> torch.allclose(img, img_ref, atol=1/65535, rtol=0)
    True
    >>> metadata == metadata_ref
    True
    >>>
    """
    assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".jp2")
    assert isinstance(image, torch.Tensor), image.__class__.__name__
    assert metadata is None or isinstance(metadata, dict), metadata.__class__.__name__

    # convertion to uint16
    img = image * 65535
    img += 0.5  # for round, not floor
    img_np = img.numpy(force=True).astype(np.uint16)

    # encode image into jpeg2000
    succes, img_data = cv2.imencode(".jp2", img_np, (cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000))
    if not succes:
        raise ValueError(f"failed to encode image {filename} with cv2")

    # encode metadata
    if metadata:  # no test 'is None' to avoid encoding empty dict
        metadata = yaml.safe_dump(metadata)  # pickle is not security safe
        metadata = lzma.compress(metadata.encode(), preset=9 | lzma.PRESET_EXTREME)
        img_data = b"".join((img_data, b"laueimproc_exif:", metadata))

    # write file
    with open(filename, "wb") as raw:
        raw.write(img_data)
