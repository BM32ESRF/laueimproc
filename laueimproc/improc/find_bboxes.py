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


def _merge_bboxes(bboxes: list[list[int]], src: int, dst: int) -> list[list[int]]:
    """Merge the bboxes inplace.

    bbox_dst = union(bbox_dst, bbox_src)
    """
    bbox_src, bbox_dst = bboxes[src-1], bboxes[dst-1]
    end = max(bbox_src[0] + bbox_src[2], bbox_dst[0] + bbox_dst[2])
    bbox_dst[0] = min(bbox_src[0], bbox_dst[0])
    bbox_dst[2] = end - bbox_dst[0]
    end = max(bbox_src[1] + bbox_src[3], bbox_dst[1] + bbox_dst[3])
    bbox_dst[1] = min(bbox_src[1], bbox_dst[1])
    bbox_dst[3] = end - bbox_dst[1]
    return bboxes


def _python_find_bboxes(binary: np.ndarray[np.uint8], max_size: numbers.Integral) -> list:
    """Help find_bboxes with a pure very slow python version."""
    curr_clus = 0
    bboxes: list[list[int]] = []
    clusters: list[int] = [0 for _ in range(binary.shape[1])]
    merge: dict[int, int] = {}

    # find clusters
    for i in range(binary.shape[0]):
        clus_left = 0
        for j in range(binary.shape[1]):
            if not binary[i, j]:  # case black pixel
                clusters[j] = 0
                clus_left = 0
                continue
            clus_top = clusters[j]
            if (not clus_top) and (not clus_left):  # case new cluster
                bboxes.append([i, j, 1, 1])
                curr_clus += 1
                clusters[j] = curr_clus  # next cluster
            elif clus_top == clus_left:  # case same clusters
                clusters[j] = clus_top
            elif not clus_top:  # case same as left
                clus_left = merge.get(clus_left, clus_left)  # avoid cycle
                right = max(j+1, bboxes[clus_left-1][1] + bboxes[clus_left-1][3])  # right pos
                bboxes[clus_left-1][3] = right - bboxes[clus_left-1][1]
                clusters[j] = clus_left
            elif not clus_left:  # case same as top
                clus_top = merge.get(clus_top, clus_top)  # avoid cycle
                bottom = max(i+1, bboxes[clus_top-1][0] + bboxes[clus_top-1][2])  # bottom pos
                bboxes[clus_top-1][2] = bottom - bboxes[clus_top-1][0]
                clusters[j] = clus_top
            else:  # case merge
                if clus_top > clus_left:  # guaranteed that clu > merge[clu-1]
                    clus_left, clus_top = clus_top, clus_left
                clus_top = merge.get(clus_top, clus_top)  # avoid cycle
                merge[clus_left] = clus_top
                clusters[j] = clus_top
                bboxes = _merge_bboxes(bboxes, clus_left, clus_top)
            clus_left = clusters[j]

    # merge clusters
    for src, dst in merge.items():  # merge
        bboxes = _merge_bboxes(bboxes, src, dst)
    bboxes = [  # concatenate
        bbox for clu_m1, bbox in enumerate(bboxes)
        if (
            clu_m1 + 1 not in merge  # skip same bboxes
            and (bbox[2] <= max_size and bbox[3] <= max_size)  # remove too tall bboxes
        )
    ]
    return bboxes


def find_bboxes(
    binary: np.ndarray[np.uint8],
    max_size: numbers.Integral = 255,
    *, use_cv2: bool = False,
    _no_c: bool = False,
) -> torch.Tensor:
    """Find the bboxes of the binary image.

    The proposed algorithm is strictely equivalent to find the clusters of the binary image
    with a DBSCAN of distance 1, then to find the bboxes of each cluster.

    Parameters
    ----------
    binary : np.ndarray
        The c contiguous 2d image of boolean.
    max_size : int
        The max bounding boxe size, reject the strictly bigger bboxes.
    use_cv2 : bool, default=False
        If set to True, use the algorithm provided by cv2 rather than laueimproc's own algorithme.
        Be carefull, behavior is not the same!
        There are a few misses and the clusters next to each other on a diagonal are merged.
        The cv2 algorithm is about 3 times slowler than the compiled C version for high density,
        but still around 100 times faster than the pure python version.
        The cv2 algorithm doesn't release the GIL, making it difficult to multithread.

    Returns
    -------
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        They are sorted by ascending colums then lignes. This order optimizes the image cache acces.

    Examples
    --------
    >>> import numpy as np
    >>> from laueimproc.improc.find_bboxes import find_bboxes
    >>> binary = np.array([[1, 0, 1, 0, 1],
    ...                    [0, 1, 0, 1, 0],
    ...                    [1, 0, 1, 0, 1]],
    ...                   dtype=np.uint8)
    >>> find_bboxes(binary)
    tensor([[0, 0, 1, 1],
            [0, 2, 1, 1],
            [0, 4, 1, 1],
            [1, 1, 1, 1],
            [1, 3, 1, 1],
            [2, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 4, 1, 1]], dtype=torch.int16)
    >>> binary = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...                    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    ...                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    ...                    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    ...                    [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ...                    [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    ...                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    ...                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
    ...                   dtype=np.uint8)
    >>> find_bboxes(binary)
    tensor([[ 0,  0,  1,  1],
            [ 0,  6,  3,  1],
            [ 0,  8,  3,  2],
            [ 0, 11,  7, 10],
            [ 1,  1,  1,  1],
            [ 1,  3,  1,  2],
            [ 3,  1,  2,  1],
            [ 3,  3,  2,  2],
            [ 4,  4,  5,  5],
            [ 6,  0,  1,  3],
            [ 8,  0,  2,  3],
            [ 8,  8,  3,  3],
            [ 8, 12,  3,  4],
            [11,  0, 10,  7],
            [12,  8,  4,  3],
            [12, 14,  2,  2],
            [14, 12,  2,  2],
            [14, 16,  2,  2],
            [16, 14,  2,  2],
            [17, 17,  4,  4]], dtype=torch.int16)
    >>> def show():
    ...     import matplotlib.pyplot as plt
    ...     plt.imshow(
    ...         binary,
    ...         extent=(0, binary.shape[1], binary.shape[0], 0),
    ...         cmap="gray",
    ...         interpolation=None,
    ...     )
    ...     bboxes = find_bboxes(binary)
    ...     plt.plot(
    ...         np.vstack((
    ...             bboxes[:, 1],
    ...             bboxes[:, 1],
    ...             bboxes[:, 1]+bboxes[:, 3],
    ...             bboxes[:, 1]+bboxes[:, 3],
    ...             bboxes[:, 1],
    ...         )),
    ...         np.vstack((
    ...             bboxes[:, 0],
    ...             bboxes[:, 0]+bboxes[:, 2],
    ...             bboxes[:, 0]+bboxes[:, 2],
    ...             bboxes[:, 0],
    ...             bboxes[:, 0],
    ...         )),
    ...     )
    ...     plt.show()
    ...
    >>> show()  # doctest: +SKIP
    >>> binary = (np.random.random((2048, 2048)) > 0.75).view(np.uint8)
    >>> np.array_equal(find_bboxes(binary), find_bboxes(binary, _no_c=True))
    True
    >>>
    """
    assert isinstance(use_cv2, bool), use_cv2.__class__.__name__

    if not use_cv2 and not _no_c and c_find_bboxes is not None:
        return torch.from_numpy(c_find_bboxes.find_bboxes(binary, max_size))

    assert isinstance(binary, np.ndarray), binary.__class__.__name__
    assert binary.ndim == 2, binary.shape
    assert binary.dtype == np.uint8, binary.dtype
    assert isinstance(max_size, numbers.Integral), max_size.__class__.__name__
    assert max_size >= 1, max_size

    if use_cv2:
        bboxes = [
            [i, j, h, w] for j, i, w, h in map(  # cv2 to numpy convention
                cv2.boundingRect,
                cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
            ) if max(h, w) <= max_size  # remove too big spots
        ]
    else:
        bboxes = _python_find_bboxes(binary, max_size)

    if bboxes:
        bboxes = torch.tensor(bboxes, dtype=torch.int16)
    else:
        bboxes = torch.empty((0, 4), dtype=torch.int16)
    return bboxes


# if __name__ == '__main__':
#     from laueimproc.improc.peaks_search import (_density_to_threshold_torch,
#         DEFAULT_KERNEL_FONT, DEFAULT_KERNEL_AGLO, estimate_background
#     )
#     from laueimproc.io import get_sample
#     from laueimproc import Diagram
#     src = Diagram(get_sample()).image.numpy(force=True)
#     bg_image = estimate_background(src, DEFAULT_KERNEL_FONT)
#     fg_image = src - bg_image
#     import timeit
#     for density in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
#         print("density", density)
#         binary = (
#             (fg_image > _density_to_threshold_torch(torch.from_numpy(fg_image), density))
#             .view(np.uint8)
#         )
#         binary = cv2.dilate(binary, DEFAULT_KERNEL_AGLO, dst=bg_image, iterations=1)
#         t1=min(timeit.repeat(lambda: find_bboxes(binary, use_cv2=True), repeat=10, number=50))/50
#         print(f"time cv2 {1000*t1:.2f}ms")
#         t2 = min(timeit.repeat(lambda: find_bboxes(binary), repeat=10, number=50))/50
#         print(f"time c   {1000*t2:.2f}ms")
#         print(f"c is {t1/t2:.1f} times faster than cv2")
#         import matplotlib.pyplot as plt
#         diagram = Diagram(get_sample())
#         diagram.find_spots(density=density)
#         fig = plt.figure()
#         diagram.plot(fig)
#         plt.show()
