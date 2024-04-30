#!/usr/bin/env python3

"""Efficient Principal Componant Annalysis on spots."""

import logging

import torch

from laueimproc.gmm.linalg import cov2d_to_eigtheta
try:
    from laueimproc.improc.spot import c_pca
except ImportError:
    logging.warning(
        "failed to import laueimproc.improc.spot.c_pca, a slow python version is used instead"
    )
    c_pca = None
from laueimproc.opti.rois import rawshapes2rois


def compute_rois_pca(data: bytearray, bboxes: torch.Tensor, *, _no_c: bool = False) -> torch.Tensor:
    r"""Compute the PCA for each spot.

    See ``laueimproc.gmm`` for terminology.

    Parameters
    ----------
    data : bytearray
        The raw data \(\alpha_i\) of the concatenated not padded float32 rois.
    bboxes : torch.Tensor
        The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width)
        for each spots, of shape (n, 4). It doesn't have to be c contiguous.

    Returns
    -------
    pca : torch.Tensor
        The concatenation of the colum vectors of the two std in pixel and the angle in radian
        \( \left[ \sqrt{\lambda_1}, \sqrt{\lambda_2}, \theta \right] \) of shape (n, 3).
        The diagonalization of \(mathbf{\Sigma}\)
        is performed with ``laueimproc.gmm.linalg.cov2d_to_eigtheta``.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from laueimproc.improc.spot.pca import compute_rois_pca
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
    >>> compute_rois_pca(
    ...     bytearray(rois.tobytes()), torch.tensor([[0, 0, 5, 5]]*4, dtype=torch.int16)
    ... )
    tensor([[0.6325, 0.0000, 0.0000],
            [0.8944, 0.0000, 0.7854],
            [0.6325, 0.0000, 1.5708],
            [0.4518, 0.2020, 0.0000]])
    >>> bboxes = torch.zeros((1000, 4), dtype=torch.int16)
    >>> bboxes[::2, 2], bboxes[1::2, 2], bboxes[::2, 3], bboxes[1::2, 3] = 10, 20, 30, 40
    >>> data = bytearray(
    ...     np.linspace(0, 1, (bboxes[:, 2]*bboxes[:, 3]).sum(), dtype=np.float32).tobytes()
    ... )
    >>> std1_std2_theta = compute_rois_pca(data, bboxes)
    >>> torch.allclose(std1_std2_theta[:, :2], compute_rois_pca(data, bboxes, _no_c=True)[:, :2])
    True
    >>>
    """
    if not _no_c and c_pca is not None:
        return torch.from_numpy(
            c_pca.compute_rois_pca(data, bboxes.numpy(force=True))
        ).to(bboxes.device)

    assert isinstance(bboxes, torch.Tensor), bboxes.__class__.__name__
    assert bboxes.ndim == 2, bboxes.shape
    assert bboxes.shape[1] == 4, bboxes.shape

    # preparation
    rois = rawshapes2rois(data, bboxes[:, 2:], _no_c=_no_c)  # (n, h, w), more verif here
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
    mass *= mass
    cov /= mass
    eig1, eig2, theta = cov2d_to_eigtheta(cov).mT  # diagonalization
    std1, std2 = torch.sqrt(eig1), torch.sqrt(eig2)  # var to std
    return torch.cat([std1.unsqueeze(1), std2.unsqueeze(1), theta.unsqueeze(1)], axis=1)
