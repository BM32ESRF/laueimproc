#!/usr/bin/env python3

"""Link serveral diagrams together."""

import numbers
import typing

import torch

from laueimproc.ml.dataset_dist import select_closest
from .base_dataset import BaseDiagramsDataset
from .diagram import Diagram


class DiagramsDataset(BaseDiagramsDataset):
    """Link Diagrams together."""

    def compute_mean_image(self) -> torch.Tensor:
        """Compute the average image.

        Based on ``laueimproc.immix.mean.mean_stack``.

        Returns
        -------
        torch.Tensor
            The average image of all the images contained in this dataset.
        """
        from laueimproc.immix.mean import mean_stack
        return mean_stack(self)

    def compute_inter_image(self, *args, method: str = "snowflake", **kwargs) -> torch.Tensor:
        """Compute the median, first quartile, third quartile or everything in between.

        Parameters
        ----------
        *args, **kwargs
            Transmitted to ``laueimproc.immix.inter.sort_stack``
            or ``laueimproc.immix.inter.snowflake_stack``.
        method : str
            The algorithm used.
            It can be `sort` for the naive accurate algorithm,
            or `snowflake` for a faster algorithm on big dataset, but less accurate.

        Returns
        -------
        torch.Tensor
            The 2d float32 grayscale image.
        """
        assert isinstance(method, str), method.__class__.__name__
        match method:
            case "snowflake":
                from laueimproc.immix.inter import snowflake_stack
                return snowflake_stack(self, *args, **kwargs)
            case "sort":
                from laueimproc.immix.inter import sort_stack
                return sort_stack(self, *args, **kwargs)
            case _:
                raise ValueError(f"method can only be 'snowflake' or 'sort', not {method}")

    def select_closest(
        self, *args, no_raise: bool = False, **kwargs
    ) -> typing.Union[Diagram, None]:
        """Select the closest diagram to a given set of phisical parameters.

        Parameters
        ----------
        point, tol, scale
            Transmitted to ``laueimproc.ml.dataset_dist.select_closest``.
        no_raise : boolean, default = False
            If set to True, return None rather throwing a LookupError exception.

        Returns
        -------
        closet_diag : ``laueimproc.classes.diagram.Diagram``
            The closet diagram.

        Examples
        --------
        >>> from laueimproc.classes.dataset import DiagramsDataset
        >>> from laueimproc.io import get_samples
        >>> def position(index: int) -> tuple[int, int]:
        ...     return divmod(index, 10)
        ...
        >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position)
        >>> dataset.select_closest((4.1, 4.9), tol=(0.2, 0.2))
        Diagram(img_45.jp2)
        >>>
        """
        assert isinstance(no_raise, bool), no_raise.__class__.__name__
        indices, coords = self.get_scalars(as_dict=False, copy=False)
        try:
            index = select_closest(coords, *args, **kwargs)
        except LookupError as err:
            if no_raise:
                return None
            raise err
        return self._get_diagram_from_index(int(indices[index]))

    def track_spots(self):
        """Find the apparition of the same spots in differents diagrams."""
        raise NotImplementedError

    def train_spot_classifier(
        self,
        *,
        model=None,
        shape: typing.Union[numbers.Real, tuple[numbers.Integral, numbers.Integral]] = 0.95,
        **kwargs,
    ):
        """Train a variational autoencoder classifier with the spots in the diagrams.

        It is a non supervised neuronal network.

        Parameters
        ----------
        model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier, optional
            If provided, this model will be trained again and returned.
        shape : float or tuple[int, int], default=0.95
            The model input spot shape in numpy convention (height, width).
            If a percentage is given, the shape is founded with ``laueimproc.nn.loader.find_shape``.
        space : float, default = 3.0
            The non penalized spreading area half size.
            Transmitted to ``laueimproc.nn.vae_spot_classifier.VAESpotClassifier``.
        intensity_sensitive, scale_sensitive : boolean, default=True
            Transmitted to ``laueimproc.nn.vae_spot_classifier.VAESpotClassifier``.
        epoch, lr, fig
            Transmitted to ``laueimproc.nn.train.train_vae_spot_classifier``.

        Returns
        -------
        classifier: laueimproc.nn.vae_spot_classifier.VAESpotClassifier
            The trained model.

        Examples
        --------
        >>> import laueimproc
        >>> def init(diagram: laueimproc.Diagram):
        ...     diagram.find_spots()
        ...     diagram.filter_spots(range(10))  # to reduce the number of spots
        ...
        >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples())
        >>> _ = dataset.apply(init)
        >>> model = dataset.train_spot_classifier(epoch=2)
        >>>
        """
        from laueimproc.nn.loader import find_shape
        from laueimproc.nn.train import train_vae_spot_classifier
        from laueimproc.nn.vae_spot_classifier import _test_size, VAESpotClassifier

        if model is None:
            if isinstance(shape, numbers.Real):
                shape = find_shape(self, shape)
            while not _test_size(shape[0]):
                shape = (shape[0]+1, shape[1])
            while not _test_size(shape[1]):
                shape = (shape[0], shape[1]+1)
            model = VAESpotClassifier(shape, latent_dim=2, **kwargs)

        assert isinstance(model, VAESpotClassifier), model.__class__.__name__
        train_vae_spot_classifier(model, self, **kwargs)

        return model
