#!/usr/bin/env python3

"""Link serveral diagrams together."""

import numbers
import typing

from laueimproc.classes.base_dataset import BaseDiagramsDataset


class DiagramsDataset(BaseDiagramsDataset):
    """Link Diagrams together."""

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
