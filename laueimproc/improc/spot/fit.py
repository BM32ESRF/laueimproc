#!/usr/bin/env python3

"""Fit the spot by one gaussian."""

import collections
import typing

import torch

from laueimproc.classes.tensor import Tensor
from laueimproc.improc.gmm import em


def fit_gaussian(
    rois: Tensor,
    photon_density: typing.Union[float, Tensor] = 1.0,
    *,
    tol: bool = False,
    **extra_info,
) -> tuple[Tensor, Tensor, dict]:
    r"""Fit each roi by one gaussian.

    See ``laueimproc.improc.gmm`` for terminology.

    Parameters
    ----------
    rois : Tensor
        The tensor of the regions of interest for each spots. Shape (n, h, w).
    photon_density : float or Tensor
        Convertion factor to transform the intensity of a pixel
        into the number of photons that hit it.
        Note that the range of intensity values is between 0 and 1.
    tol : boolean, default=False
        If set to True, the accuracy measure is happend into `infodict`. Shape (n,).
        It correspond to the standard deviation of the estimation of the mean.
        Let \(\widehat{\mathbf{\mu}}\) be the estimator of the mean.

        * \(\begin{cases}
            std(\widehat{\mathbf{\mu}}) = \sqrt{var(\widehat{\mathbf{\mu}})} \\
            var(\widehat{\mathbf{\mu}}) = var\left( \frac{
                    \sum\limits_{i=1}^N \alpha_i \mathbf{x}_i
                }{
                    \sum\limits_{i=1}^N \alpha_i
                } \right) = \frac{
                    \sum\limits_{i=1}^N var(\alpha_i \mathbf{x}_i)
                }{
                    \left( \sum\limits_{i=1}^N \alpha_i \right)^2
                } \\
            var(\alpha_i \mathbf{x}_i) = var(
                \mathbf{x}_{i1} + \mathbf{x}_{i2} + \dots + \mathbf{x}_{i\alpha_i}) = \alpha_i var(
                \mathbf{x}_i) && \text{because } \mathbf{x}_{ij} \text{ are i.i.d photons} \\
            var(\mathbf{x}_i) = max(eigen( \mathbf{\Sigma} )) \\
        \end{cases}\)
        * \(
            var(\widehat{\mathbf{\mu}})
            = \frac{ max(eigen(\mathbf{\Sigma})) }{ \sum\limits_{i=1}^N \alpha_i }
        \)
        * \(
            std(\widehat{\mathbf{\mu}})\
            = \sqrt{ \frac{ max(eigen(\mathbf{\Sigma})) }{ \sum\limits_{i=1}^N \alpha_i } }
        \)
    **extra_infos : dict
        See ``laueimproc.improc.gmm.em`` for the available metrics.

    Returns
    -------
    mean : Tensor
        The vectors \(\mathbf{\mu}\). Shape (n, \(D\), 1). In the relative roi base.
    cov : Tensor
        The matrices \(\mathbf{\Sigma}\). Shape (n, \(D\), \(D\)).
    infodict : dict[str]
        A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
    """
    # verification
    assert isinstance(rois, Tensor), rois.__class__.__name__
    assert rois.ndim == 3, rois.shape
    assert isinstance(photon_density, (float, Tensor)), photon_density.__class__.__name__
    if isinstance(photon_density, Tensor):
        assert photon_density.shape == (rois.shape[0],), photon_density.shape
    assert isinstance(tol, bool), tol.__class__.__name__

    # preparation
    points_i, points_j = torch.meshgrid(
        torch.arange(0.5, rois.shape[1]+0.5, dtype=rois.dtype, device=rois.device),
        torch.arange(0.5, rois.shape[2]+0.5, dtype=rois.dtype, device=rois.device),
        indexing="ij",
    )
    points_i, points_j = points_i.ravel(), points_j.ravel()
    obs = torch.cat([points_i.unsqueeze(-1), points_j.unsqueeze(-1)], axis=1)
    obs = obs.expand(rois.shape[0], -1, -1)  # (n_spots, n_obs, n_var)
    dup_w = torch.reshape(rois, (rois.shape[0], -1))
    dup_w = dup_w * photon_density  # copy (no inplace) for keeping rois unchanged

    # fit gaussian
    mean, cov, _, infodict = em(obs, dup_w, **extra_info, nbr_clusters=1)

    # estimation of mean tolerancy
    if tol:
        std_of_mean = torch.max(torch.linalg.eigvalsh(cov), axis=-1).values.squeeze(-1)
        std_of_mean /= torch.sum(dup_w, axis=-1)
        std_of_mean = torch.sqrt(std_of_mean, out=std_of_mean)
        infodict["tol"] = std_of_mean

    # cast
    return collections.namedtuple(
        "FitGaussian", ("mean", "cov", "infodict")
    )(mean.squeeze(-3), cov.squeeze(-3), infodict)
