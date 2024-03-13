#!/usr/bin/env python3

"""Define the data sructure of a single Laue diagram image."""

import collections
import numbers
import typing

from matplotlib.axes import Axes
import numpy as np
import torch

from laueimproc.improc.spot.basic import compute_barycenters, compute_pxl_intensities
from laueimproc.improc.spot.fit import fit_gaussian
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
    def fit_gaussian(
        self,
        photon_density: typing.Union[torch.Tensor, np.ndarray, numbers.Real] = 1.0,
        **kwargs,
    ) -> tuple[Tensor, Tensor, dict]:
        r"""Fit each roi by one gaussian.

        See ``laueimproc.improc.gmm`` for terminology.

        Parameters
        ----------
        photon_density : arraylike, optional
            See to ``laueimproc.improc.spot.fit.fit_gaussian``.
        **kwargs : dict
            Transmitted to ``laueimproc.improc.spot.fit.fit_gaussian``.

        Returns
        -------
        mean : Tensor
            The vectors \(\mathbf{\mu}\). Shape (..., \(D\), 1). In the absolute diagram base.
        cov : Tensor
            The matrices \(\mathbf{\Sigma}\). Shape (..., \(D\), \(D\)).
        infodict : dict[str]
            A dictionary of optional outputs (see ``laueimproc.improc.gmm.em``).
        """
        photon_density = (
            float(photon_density)
            if isinstance(photon_density, numbers.Real)
            else Tensor(photon_density)
        )

        # preparation
        if self.spots is None:
            raise RuntimeWarning(
                "you must to initialize the spots (`self.find_spots()`)"
            )
        rois = self.rois
        shift = self.bboxes[:, :2]

        # main fit
        mean, cov, infodict = fit_gaussian(rois, photon_density, **kwargs)

        # spot base to diagram base
        if mean.requires_grad:
            mean = mean + shift.unsqueeze(-1)
        else:
            mean += shift.unsqueeze(-1)

        # cast
        return collections.namedtuple(
            "FitGaussian", ("mean", "cov", "infodict")
        )(mean.squeeze(-3), cov.squeeze(-3), infodict)


    def plot(
        self,
        *args,
        confidence: typing.Optional[typing.Union[numbers.Real, np.ndarray, torch.tensor]] = None,
        **kwargs,
    ) -> Axes:
        """Add gaussian fit informations.

        Paremeters
        ----------
        *args, **kwargs
            Transmitted to ``laueimproc.classes.base_diagram.BaseDiagram.plot``.
        confidence : arraylike, optional
            The vector of the interval of confidence of the position.
        """
        if confidence is not None:
            if isinstance(confidence, numbers.Real):
                confidence = float(confidence)
                assert confidence >= 0, confidence
            else:
                confidence = Tensor(confidence).squeeze()
                assert confidence.ndim == 1, confidence.shape
                assert torch.all(confidence >= 0), confidence

        axes = super().plot(*args, **kwargs)

        # baricenters
        if self.compute_barycenters(is_cached=True) or confidence is not None:
            mean = self.compute_barycenters()
            mean = torch.transpose(mean, 0, 1)
            axes.scatter(*mean.numpy(force=True), color="blue", marker="x")

        # confidence
        if confidence is not None:
            circles = torch.linspace(0, 2*torch.pi, 33, dtype=mean.dtype, device=mean.device)
            circles = torch.cat(
                [torch.cos(circles).reshape(1, -1, 1), torch.sin(circles).reshape(1, -1, 1)],
                axis=0,
            )
            if isinstance(confidence, Tensor):
                circles = confidence.reshape(1, 1, -1) * circles
            else:
                circles *= confidence
            circles = mean.unsqueeze(1) + circles
            axes.plot(*circles.numpy(force=True), color="green", scalex=False, scaley=False)

        return axes
