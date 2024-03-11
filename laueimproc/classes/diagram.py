#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import torch

from laueimproc.improc.gmm import em
from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.opti.cache import auto_cache
from laueimproc.opti.parallel import auto_parallel
from .base_diagram import BaseDiagram
from .tensor import Tensor


class Diagram(BaseDiagram):
    """A Laue diagram image."""

    @auto_cache  # put the result in thread safe cache (no multiprocessing)
    @auto_parallel  # automaticaly multithreading
    def compute_barycenters(self) -> Tensor:
        """Compute the barycenter of each spots."""
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        barycenters = compute_barycenters(self.rois)  # relative to each spots
        barycenters += self.bboxes[:, :2].to(barycenters.dtype)  # absolute
        return barycenters

    @auto_cache
    @auto_parallel
    def compute_pxl_intensities(self) -> Tensor:
        """Compute the total pixel intensity for each spots."""
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        return compute_pxl_intensities(self.rois)

    @auto_cache
    @auto_parallel
    def fit_gaussian(self, **metrics) -> tuple[Tensor, Tensor, dict]:
        r"""Fit each roi by one gaussian.

        See ``laueimproc.improc.gmm`` for terminology.

        Parameters
        ----------
        **extra_infos : dict
            See ``laueimproc.improc.gmm.em`` for the available metrics.

        Returns
        -------
        mean : Tensor
            The vectors \(\mathbf{\mu}\). Shape (..., \(D\), 1).
        cov : Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (..., \(D\), \(D\)).
        infodict : dict[str]
            A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
        """
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        rois = self.rois
        points_i, points_j = torch.meshgrid(
            torch.arange(rois.shape[1], dtype=rois.dtype, device=rois.device),
            torch.arange(rois.shape[2], dtype=rois.dtype, device=rois.device),
            indexing="ij",
        )
        points_i, points_j = points_i.ravel(), points_j.ravel()
        obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], axis=1)
        obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
        dup_w = torch.reshape(rois, (rois.shape[0], -1))
        mean, cov, _, infodict = em(obs, dup_w, **metrics, nbr_clusters=1, _check=False)
        mean, cov = mean.squeeze(-3), cov.squeeze(-3)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + self.bboxes[:, :2].unsqueeze(-1)
        else:
            mean += self.bboxes[:, :2].unsqueeze(-1)
        return mean, cov, infodict
