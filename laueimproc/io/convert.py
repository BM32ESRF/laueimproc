#!/usr/bin/env python3

"""Convert the image file format."""

import functools
import multiprocessing.pool
import pathlib
import typing

import cv2
from tqdm.autonotebook import tqdm


PATHLIKE = typing.Union[pathlib.Path, str]
SUPPORTED_FORMATS = {
    ".bmp", ".dib",
    ".jp2",
    ".mccd",
    ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
    ".tif", ".tiff",
}


def _unfold(files_or_dirs: PATHLIKE) -> list[pathlib.Path]:
    files_or_dirs = pathlib.Path(files_or_dirs)
    if files_or_dirs.is_file():
        if files_or_dirs.suffix.lower() in SUPPORTED_FORMATS:
            return [files_or_dirs]
    elif files_or_dirs.is_dir():
        return [f for fd in files_or_dirs.iterdir() for f in _unfold(fd)]
    return []


def converter_decorator(func: typing.Callable):
    """Decorate an image converter into a batch image converter."""
    @functools.wraps(func)
    def batch_converter(
        src: typing.Union[PATHLIKE, typing.Iterable[PATHLIKE]],
        dst_dir: typing.Optional[PATHLIKE] = None,
    ):
        """Decorate an image converter.

        Parameters
        ----------
        src : pathlike or iterable of pathlike
            The files to be converted.
        dst_dir : dirlike, optional
            The output directory. By default the same directory as the input file is used.
        """
        if isinstance(src, str):
            src = pathlib.Path(src)
        if isinstance(src, pathlib.Path):
            src = _unfold(src)
        elif hasattr(src, "__iter__"):
            src = [f for fd in src for f in _unfold(fd)]
        else:
            raise TypeError(f"only pathlike or iterable supproted, not {src.__class__.__name__}")
        if dst_dir is not None:
            assert isinstance(dst_dir, (pathlib.Path, str)), dst_dir.__class__.__name__
            dst_dir = pathlib.Path(dst_dir)
            assert dst_dir.is_dir(), f"the output directpty {dst_dir} has to exists"

        file_log = tqdm(total=0, position=1, bar_format="{desc}")
        with multiprocessing.pool.ThreadPool() as pool:
            for dst_file in tqdm(
                pool.imap_unordered(lambda s: func(s, dst_dir), src),
                total=len(src),
                desc="convert",
                unit="img",
            ):
                file_log.set_description_str(f"file {dst_file} created")
    return batch_converter


@converter_decorator
def to_jp2(src_file: pathlib.Path, dst_dir: typing.Optional[pathlib.Path] = None) -> pathlib.Path:
    """Convert a file into jpeg2000 format.

    Parameters
    ----------
    src_file : pathlibPath
        The input filename.
    dst_dir : pathlib.Path
        The output directory.

    Returns
    -------
    abs_out_path : pathlib.Path
        The absolute filename of the created image.
    """
    if (img_np := cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)) is None:
        raise ValueError(f"failed to decode image {src_file} with cv2")
    if dst_dir is None:
        dst_file = src_file.with_suffix(".jp2").resolve()
    else:
        dst_file = (dst_dir / src_file.with_suffix(".jp2").name).resolve()
    if not cv2.imwrite(str(dst_file), img_np, (cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000)):
        raise ValueError(f"failed to encode image {dst_file} with cv2")
    return dst_file
