#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import torch

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.improc.spot.gmm import em
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
    def fit_gaussians(self, **kwargs) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Fit each roi by gaussians.

        See ``laueimproc.improc.spot.gmm.em`` for accurate signature.
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
        dup_w = torch.flatten(rois, start_dim=1)  # TODO opti with no copy function   # (n_spots, n_pxl)
        std_w = torch.ones_like(dup_w)  # TODO make this weight optional   # (n_spots, n_pxl)
        return em(obs, std_w, dup_w, **kwargs)

        # from laueimproc.improc.spot.gmm import em, _gauss, _bic, _aic
        # rois = self.rois
        # points_i, points_j = torch.meshgrid(
        #     torch.arange(rois.shape[1], dtype=torch.float32),
        #     torch.arange(rois.shape[2], dtype=torch.float32),
        #     indexing="ij",
        # )
        # points_i, points_j = points_i.ravel(), points_j.ravel()
        # obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], axis=1)
        # obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
        # dup_w = torch.flatten(rois, start_dim=1)  # TODO opti with no copy function   # (n_spots, n_pxl)
        # std_w = torch.ones_like(dup_w)  # TODO make this weight optional   # (n_spots, n_pxl)

        # mean, cov, eta = em(obs, std_w, dup_w, nbr_clusters=nbr_clusters)

        # return mean, cov, eta

        # # mean.requires_grad = True
        # prob = _gauss(obs, mean, cov)  # (n_spots, n_clu, n_pxl)
        # prob *= eta.unsqueeze(-1)  # relative proba
        # mass = torch.sum(rois, axis=(1, 2)).unsqueeze(-1)  # (n_spots, 1)
        # surface = torch.sum(prob, axis=1)  # (n_spots, n_pxl)
        # surface *= mass

        # mse = torch.sum((surface - dup_w)**2, axis=-1)  # (n_spots,)

        # bic = _bic(obs, dup_w, mean, cov, eta)  # (n_spots,)
        # aic = _aic(obs, dup_w, mean, cov, eta)  # (n_spots,)

        # mean = mean + self.bboxes[:, :2].unsqueeze(1).unsqueeze(3)

        # return mean, cov, eta, mse, bic, aic

