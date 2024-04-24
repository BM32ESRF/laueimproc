#!/usr/bin/env python3

"""Efficient Principal Componant Annalysis on spots."""

import logging

import numpy as np
import torch

from laueimproc.gmm.linalg import cov2d_to_eigtheta
try:
    from laueimproc.improc.spot import c_pca
except ImportError:
    logging.warning(
        "failed to import laueimproc.spot.c_pca, a slow python version is used instead"
    )
    c_pca = None
from laueimproc.opti.rois import rawshapes2rois


def pca(data: bytearray, shapes: np.ndarray[np.int32], *, _no_c: bool = False) -> torch.Tensor:
    r"""Compute the PCA for each spot.

    See ``laueimproc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data of the concatenated not padded float32 rois.
        It corresponds the the folded \(\alpha_i\).
    shapes : np.ndarray[np.int32]
        Contains the information of the bboxes shapes.
        heights = shapes[:, 0] and widths = shapes[:, 1].
        It doesn't have to be c contiguous.

    Returns
    -------
    pca : torch.Tensor
        The concatenation of the colum vectors of the two std in pixel and the angle in radian
        \( \left[ \sqrt{\lambda_1}, \sqrt{\lambda_2}, \theta \right] \) of shape (n, 3).
        The diagonalization of \(mathbf{\Sigma}\)
        is performed with ``laueimproc.gmm.linalg.cov2d_to_eigtheta``.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from laueimproc.improc.spot.pca import pca
    >>> shapes = np.zeros((1000, 2), dtype=np.int32)
    >>> shapes[::2, 0], shapes[1::2, 0], shapes[::2, 1], shapes[1::2, 1] = 10, 20, 30, 40
    >>> data = bytearray(
    ...     np.linspace(0, 1, (shapes[:, 0]*shapes[:, 1]).sum(), dtype=np.float32).tobytes()
    ... )
    >>> std1_std2_theta = pca(data, shapes)
    >>> assert torch.allclose(std1_std2_theta[:, :2], pca(data, shapes, _no_c=True)[:, :2])
    >>>
    >>> rois = np.zeros((4, 5, 5), np.float32)
    >>> rois[0, :, 2] = 1
    >>> rois[1, range(5), range(5)] = 1
    >>> rois[2, 2, :] = 1
    >>> rois[3, :, 2] = 1
    >>> rois[3, 2, 1:4] = 1
    >>> rois
    array([[[0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.]],
    <BLANKLINE>
           [[1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]],
    <BLANKLINE>
           [[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]],
    <BLANKLINE>
           [[0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.]]], dtype=float32)
    >>> pca(bytearray(rois.tobytes()), np.array([[5, 5]]*4, dtype=np.int32))
    tensor([[0.6325, 0.0000, 0.0000],
            [0.8944, 0.0000, 0.7854],
            [0.6325, 0.0000, 1.5708],
            [0.4518, 0.2020, 0.0000]])
    >>>
    """
    if not _no_c and c_pca is not None:
        return torch.from_numpy(c_pca.pca(data, shapes))

    # preparation
    rois = rawshapes2rois(data, shapes, _no_c=_no_c)  # verifs here
    points_i, points_j = torch.meshgrid(
        torch.arange(rois.shape[1], dtype=rois.dtype, device=rois.device),
        torch.arange(rois.shape[2], dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(1), points_j.unsqueeze(1)], axis=1)  # (n_spots, 2)
    obs = obs.unsqueeze(0)  # (1, n_spots, 2)
    dup_w = torch.reshape(rois, (rois.shape[0], rois.shape[1]*rois.shape[2]))  # (batch, n_spots)
    dup_w = dup_w.unsqueeze(2)  # (batch, n_spots, 1)

    # compute pca
    mass = torch.sum(dup_w, dim=1, keepdim=True)  # (batch, 1, 1)
    mean = torch.sum(dup_w*obs, dim=1, keepdim=True) / mass  # (batch, 1, 2)
    obs = dup_w * (obs - mean)  # centered and weighted
    cov = obs.mT @ obs  # (batch, 2, 2)
    cov /= mass**2
    eig1, eig2, theta = cov2d_to_eigtheta(cov).mT  # diagonalization
    std1, std2 = torch.sqrt(eig1), torch.sqrt(eig2)  # var to std
    return torch.cat([std1.unsqueeze(1), std2.unsqueeze(1), theta.unsqueeze(1)], axis=1)
