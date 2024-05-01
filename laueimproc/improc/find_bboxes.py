#!/usr/bin/env python3

"""Find the bboxes of a binary image."""

import logging
import numbers

import cv2
import numpy as np
import torch

try:
    from laueimproc.improc import c_find_bboxes
except ImportError:
    logging.warning(
        "failed to import laueimproc.improc.c_find_bboxes, a slow python version is used instead"
    )
    c_find_bboxes = None


def find_bboxes(
    binary: np.ndarray[np.uint8], max_size: numbers.Integral = 255, *, _no_c: bool = False
) -> torch.Tensor:
    """Find the bboxes of the binary image.

    Parameters
    ----------
    binary : np.ndarray
        The c contiguous 2d image of boolean.
    max_size : int
        The max bounding boxe size, reject the strictly bigger bboxes.

    Returns
    -------
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)


    Examples
    --------
    >>> from pprint import pprint
    >>> import numpy as np
    >>> from laueimproc.improc.find_bboxes import find_bboxes
    >>> binary = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    ...                   dtype=np.uint8)
    >>> bboxes = find_bboxes(binary)
    >>> pprint(sorted(map(tuple, bboxes.tolist())))
    [(0, 6, 3, 1),
     (0, 8, 3, 2),
     (1, 1, 1, 1),
     (1, 3, 1, 2),
     (3, 1, 2, 1),
     (3, 3, 6, 6),
     (6, 0, 1, 3),
     (8, 0, 2, 3)]
    >>>
    """
    if not _no_c and c_find_bboxes is not None:
        return torch.from_numpy(c_find_bboxes.find_bboxes(binary, max_size))

    assert isinstance(binary, np.ndarray), binary.__class__.__name__
    assert binary.ndim == 2, binary.shape
    assert binary.dtype == np.uint8, binary.dtype
    assert isinstance(max_size, numbers.Integral), max_size.__class__.__name__
    assert max_size >= 1, max_size

    bboxes = [
        (i, j, h, w) for j, i, w, h in map(  # cv2 to numpy convention
            cv2.boundingRect,
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
        ) if max(h, w) <= max_size  # remove too big spots
    ]
    if bboxes:
        bboxes = torch.tensor(bboxes, dtype=torch.int16)
    else:
        bboxes = torch.empty((0, 4), dtype=torch.int16)
    return bboxes


if __name__ == '__main__':
    import torch
    from laueimproc.io import get_sample
    from laueimproc.io.read import read_image
    from laueimproc.improc.peaks_search import (
        estimate_background, _density_to_threshold_torch, DEFAULT_KERNEL_FONT, DEFAULT_KERNEL_AGLO
    )
    brut_image = read_image(get_sample())
    src = brut_image.numpy(force=True)
    bg_image = estimate_background(src, DEFAULT_KERNEL_FONT)
    fg_image = src - bg_image

    for density in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print(f"density: {density}")
        binary = (
            (fg_image > _density_to_threshold_torch(torch.from_numpy(fg_image), density)).view(np.uint8)
        )
        binary = cv2.dilate(binary, DEFAULT_KERNEL_AGLO, dst=bg_image, iterations=1)

        import timeit
        t1 = min(timeit.repeat(lambda: find_bboxes(binary, _no_c=True), repeat=20, number=10))
        print(f"time cv2 {1000*t1/10:.2f}ms")
        bboxes = find_bboxes(binary, _no_c=False)
        t2 = min(timeit.repeat(lambda: find_bboxes(binary, _no_c=False), repeat=20, number=10))
        print(f"time c {1000*t2/10:.2f}ms")
        print(f"the c version is {t1/t2:.2f} time faster than the cv2 one")
        print(f"bbox mean: {bboxes.to(float).mean(dim=0)}")

        # import matplotlib.pyplot as plt
        # plt.imshow(
        #     binary.transpose(),
        #     cmap="gray",
        #     aspect="equal",
        #     extent=(0, binary.shape[0], binary.shape[1], 0),
        #     interpolation=None,  # antialiasing is True
        # )
        # bboxes = find_bboxes(binary, _no_c=True)
        # plt.plot(
        #     np.vstack((
        #         bboxes[:, 0],
        #         bboxes[:, 0]+bboxes[:, 2],
        #         bboxes[:, 0]+bboxes[:, 2],
        #         bboxes[:, 0],
        #         bboxes[:, 0],
        #     )),
        #     np.vstack((
        #         bboxes[:, 1],
        #         bboxes[:, 1],
        #         bboxes[:, 1]+bboxes[:, 3],
        #         bboxes[:, 1]+bboxes[:, 3],
        #         bboxes[:, 1],
        #     )),
        #     color="blue",
        #     scalex=False,
        #     scaley=False,
        #     alpha=0.5,
        # )
        # bboxes = find_bboxes(binary, _no_c=False)
        # plt.plot(
        #     np.vstack((
        #         bboxes[:, 0],
        #         bboxes[:, 0]+bboxes[:, 2],
        #         bboxes[:, 0]+bboxes[:, 2],
        #         bboxes[:, 0],
        #         bboxes[:, 0],
        #     )),
        #     np.vstack((
        #         bboxes[:, 1],
        #         bboxes[:, 1],
        #         bboxes[:, 1]+bboxes[:, 3],
        #         bboxes[:, 1]+bboxes[:, 3],
        #         bboxes[:, 1],
        #     )),
        #     color="red",
        #     scalex=False,
        #     scaley=False,
        #     alpha=0.5,
        # )
        # plt.show()
