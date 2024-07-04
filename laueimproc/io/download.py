#!/usr/bin/env python3

"""Download dataset or models from internet."""

import lzma
import pathlib
import tarfile
import typing
import urllib.parse
import urllib.request

from tqdm.autonotebook import tqdm


DEFAULT_FOLDER = pathlib.Path("~/.cache/laueimproc/").expanduser()


def _download(url: str):
    """Dowload and yield the brut data incrementaly."""
    if (filename := pathlib.Path(urllib.parse.urlparse(url).path).name):
        yield filename
    with urllib.request.urlopen(url) as response:
        if not filename:
            yield (filename := response.headers.get_filename())
        with tqdm(
            desc=f"Downloading {filename}",
            total=response.length,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress_bar:
            for data in response:
                yield data
                progress_bar.update(len(data))


def _decompress(suffix: str, data_gen: typing.Iterator):
    """Decompress and yield the decompressed data incrementaly."""
    if suffix in {".xz", ".lzma"}:
        decompressor = lzma.LZMADecompressor()
        for data in data_gen:
            yield decompressor.decompress(data)
    else:
        raise NotImplementedError(f"compressed format {suffix} is unknowed")


def download(
    url: str,
    folder: str | pathlib.Path = DEFAULT_FOLDER,
    force_download: bool = False,
) -> pathlib.Path:
    """Download, decompress and unpack the data given by the url.

    Parameters
    ----------
    url : str
        The internet link to download the file directely.
    folder : pathlike, optional
        The folder to store the data.
    force_download : boolean, default=False
        If set to True, download even if the data are stored localy.

    Returns
    -------
    pathlib.Path
        The path of the final data.
    """
    assert isinstance(url, str), url.__class__.__name__
    assert isinstance(folder, (str, pathlib.Path)), folder.__class__.__name__
    assert isinstance(force_download, bool), force_download.__class__.__name__

    # prepare env
    folder = pathlib.Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)  # create the folder if it dosent exists
    loader = iter(_download(url))
    file_path = folder / pathlib.Path(next(loader))

    # download and decompress
    if file_path.suffix in {".xz", ".lzma"}:
        loader = _decompress(file_path.suffix, loader)
        file_path = file_path.with_suffix("")
    if force_download or not file_path.exists():
        with open(file_path, "wb") as dst:
            for data in loader:
                dst.write(data)

    # extract from archive
    if file_path.suffix in {".tar"}:
        folder = file_path.with_suffix("")
        if force_download or not folder.exists():
            with tarfile.open(file_path) as tarball:
                tarball.extractall(
                    folder,
                    members=tqdm(
                        tarball,
                        total=len(tarball.getmembers()),
                        desc=f"Extract {file_path.name}",
                        unit="file",
                        dynamic_ncols=True,
                    ),
                )
        return folder

    return file_path


def get_samples(**kwargs) -> pathlib.Path:
    """Download the samples of laue diagram and return the folder.

    Parameters
    ----------
    **kwargs : dict
        Transmitted to ``download``.

    Returns
    -------
    folder : pathlib.Path
        The folder containing all the samples.
    """
    url = (
        "https://www.dropbox.com/scl/fi/1x0326b16k5a9j8e71arq/samples.tar.xz?"
        "rlkey=b0cnw1jfqzm38on4aodt2vwvc&dl=1"
    )
    return download(url, **kwargs)
