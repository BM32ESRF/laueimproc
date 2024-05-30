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
        The concatenation of the colum vectors of the std in pixel and the angle in radian
        \( \left[ \sqrt{\lambda_1}, \sqrt{\lambda_2}, \theta \right] \) of shape (n, 3).
        The diagonalization of \(\mathbf{\Sigma}\)
        is performed with ``laueimproc.gmm.linalg.cov2d_to_eigtheta``.

    Examples
    --------
    >>> from math import cos, sin
    >>> import numpy as np
    >>> import torch
    >>> from laueimproc.improc.spot.pca import compute_rois_pca
    >>>
    >>> lambda1, lambda2, theta = 100**2, 50**2, 0.3
    >>> rot = torch.asarray([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    >>> diag = torch.asarray([[lambda1, 0.0], [0.0, lambda2]])
    >>> cov = rot @ diag @ rot.mT
    >>> np.random.seed(0)
    >>> points = torch.from_numpy(np.random.multivariate_normal([0, 0], cov, 1_000_000)).to(int)
    >>> points -= points.amin(dim=0).unsqueeze(0)
    >>> height, width = int(points[:, 0].max() + 1), int(points[:, 1].max() + 1)
    >>> rois = points[:, 0]*width + points[:, 1]
    >>> rois = torch.bincount(rois, minlength=height*width).to(torch.float32)
    >>> rois /= rois.max()  # in [0, 1]
    >>> rois = rois.reshape(1, height, width)
    >>> compute_rois_pca(
    ...     bytearray(rois.numpy().tobytes()),
    ...     torch.asarray([[0, 0, height, width]], dtype=torch.int16)
    ... )
    tensor([[99.4283, 49.6513,  0.2973]])
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
    >>> data = bytearray(rois.tobytes())
    >>> bboxes = torch.asarray([[0, 0, 5, 5]]*4, dtype=torch.int16)
    >>> compute_rois_pca(data, bboxes)
    tensor([[1.4142, 0.0000, 0.0000],
            [2.0000, 0.0000, 0.7854],
            [1.4142, 0.0000, 1.5708],
            [1.1952, 0.5345, 0.0000]])
    >>>
    >>> torch.allclose(_, compute_rois_pca(data, bboxes, _no_c=True))
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
    obs = (obs - mean).unsqueeze(-1)  # (batch, n_spots, 2, 1)
    cov = torch.sum(  # (..., 2, 2)
        dup_w.unsqueeze(3) * obs @ obs.mT,
        dim=1, keepdim=False  # (..., n_spots, 2, 2) -> (..., 2, 2)
    )
    cov /= mass
    eig1, eig2, theta = cov2d_to_eigtheta(cov).mT  # diagonalization
    std1, std2 = torch.sqrt(eig1), torch.sqrt(eig2)  # var to std
    return torch.cat([std1.unsqueeze(1), std2.unsqueeze(1), theta.unsqueeze(1)], axis=1)
