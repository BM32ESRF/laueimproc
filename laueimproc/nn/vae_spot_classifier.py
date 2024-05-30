#!/usr/bin/env python3

"""Classifier of laue spots using variational convolutive auto-encoder."""

import math
import numbers
import typing
import warnings

# from tqdm.autonotebook import tqdm
import torch


def _test_size(size: int) -> bool:
    """Return True if the size is ok."""
    if 3 <= size <= 32:
        return True
    size, rest = divmod(size, 2)
    if rest:
        return False
    return _test_size(size)


class VAESpotClassifier(torch.nn.Module):
    """A partialy convolutive variationel auto encoder used for unsupervised spot classification.

    Attributes
    ----------
    decoder : Decoder
        The decoder part, able to random draw and reconstitue an image from the encoder.
    device: torch.device
        The device of the model.
    encoder : Encoder
        The encoder part, able to transform an image into au gaussian law.
    latent_dim : int
        The dimension of the latent space.
    shape : tuple[int, int]
        The shape of the rois.
    space : float, default = 3.0
        The non penalized spreading area half size.
    """

    def __init__(
        self,
        shape: typing.Container[numbers.Integral],
        latent_dim: numbers.Integral = 2,
        space: numbers.Real = 3.0,
        **kwargs,
    ):
        """Initialise the model.

        Parameters
        ----------
        shape : tuple[int, int]
            Transmitted to ``laueimproc.nn.vae_spot_classifier.Encoder``
            and ``laueimproc.nn.vae_spot_classifier.Decoder``.
        latent_dim : int, default=2
            Transmitted to ``laueimproc.nn.vae_spot_classifier.Encoder``
            and ``laueimproc.nn.vae_spot_classifier.Decoder``.
        space : float, default=3.0
            The non penalized spreading area in the latent space.
            All the points with abs(p) <= space are autorized.
            A small value condensate all the data, very continuous space but hard to split.
            In a other way, a large space split the clusters but the values betwean the clusters
            are not well defined.
        intensity_sensitive : boolean, default=True
            If set to False, the model will not consider the spots intensity,
            as they will be normalized to have a power of 1.
        scale_sensitive : boolean = True
            If set to False, the model will not consider the spots size,
            as they will be resized and reinterpolated to a constant shape.
        """
        assert hasattr(shape, "__iter__"), shape.__class__.__name__
        shape = tuple(shape)
        assert len(shape) == 2, shape
        assert isinstance(shape[0], numbers.Integral) and isinstance(shape[1], numbers.Integral), \
            shape
        shape = (int(shape[0]), int(shape[1]))
        assert isinstance(latent_dim, numbers.Integral), latent_dim.__class__.__name__
        assert latent_dim >= 1, latent_dim
        latent_dim = int(latent_dim)
        assert isinstance(space, numbers.Real), space.__class__.__name__
        assert space > 0, space
        space = float(space)
        self._latent_dim = latent_dim
        self._shape = shape
        self._space = space
        self._sensitivity = {}
        self._sensitivity["intensity"] = kwargs.get("intensity_sensitive", True)
        assert isinstance(self._sensitivity["intensity"], bool), self._sensitivity["intensity"]
        self._sensitivity["scale"] = kwargs.get("scale_sensitive", True)
        assert isinstance(self._sensitivity["scale"], bool), self._sensitivity["scale"]

        self._normalization = None

        super().__init__()
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def dataaug(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all the data augmentations on the image.

        Parameters
        ----------
        image : torch.Tensor
            The image of shape (h, w).

        Returns
        -------
        aug_batch : torch.Tensor
            The augmented stack of images of shape (n, 1, h', w').
        """
        assert isinstance(image, torch.Tensor), image.__class__.__name__
        assert image.ndim == 2 and image.shape >= (1, 1), image.shape

        # reshape or pad/crop
        if self._sensitivity["scale"]:
            from laueimproc.nn.dataaug.scale import rescale
            image = rescale(image, self.shape, copy=False)
        else:
            from laueimproc.nn.dataaug.patch import patch
            image = patch(image, self.shape, copy=False)

        # intensity normalization
        if not self._sensitivity["intensity"]:
            energy = torch.sqrt(torch.sum(image * image))
            image = image / (energy + torch.finfo(image.dtype).eps)

        return image

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def forward(self, data) -> torch.Tensor:
        """Encode, random draw and decode the image.

        Parameters
        ----------
        data : laueimproc.classes.base_diagram.BaseDiagram or torch.Tensor
            If a digram is provided, the spots are extract, data augmentation are applied,
            and the mean projection in the latent space is returned.
            If the input is a tensor, data augmentation are not applied.
            Return the autoencoded data, after having decoded a random latent space draw.

        Returns
        -------
        torch.Tensor
            The mean latent vector of shape (n, latent_dim) if the input is a Diagram.
            The generated image of shape (n, height, width) otherwise.
        """
        from laueimproc.classes.base_diagram import BaseDiagram
        assert isinstance(data, (torch.Tensor, BaseDiagram)), data.__class__.__name__

        # case tensor
        if isinstance(data, torch.Tensor):
            if data.ndim == 2:
                return self.forward(data.unsqueeze(0)).squeeze(0)
            assert data.ndim == 3, data.shape
            mean, std = self.encoder(data.unsqueeze(1))
            sample = self.decoder.parametrize(mean, std) if self.training else mean
            generated_image = self.decoder(sample)
            return generated_image.squeeze(1)

        # case diagram
        batch = torch.empty((len(data), *self.shape), dtype=torch.float32, device=self.device)
        for i, (roi, (height, width)) in enumerate(zip(data.rois, data.bboxes[:, 2:].tolist())):
            batch[i] = self.dataaug(roi[:height, :width])
        mean, _ = self.encoder(batch.unsqueeze(1))
        return mean

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim

    def loss(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward the data and compute the loss values.

        Parameters
        ----------
        batch : torch.Tensor
            The image stack of shape (n, h, w).

        Returns
        -------
        mse_loss : torch.Tensor
            The sum of the mean square error loss for each image in the batch, shape (1,).
        kld_loss : torch.Tensor
            Pretty close to the sum of the Kullback-Leibler divergence
            for each projection in the batch, shape (1,).
            It is not litteraly the Kullback-Leibler divergence because the peanality for the mean
            is less strict, the cost is 0 in the [-space, space] interval.
            The cost is minimum when var=1 and -space<=mean<=space.

        Notes
        -----
        * No verifications are performed for performance reason.
        * The reduction is sum and not mean because it ables to split the batch in several slices.
        """
        assert isinstance(batch, torch.Tensor), batch.__class__.__name__
        assert batch.ndim == 3, batch.shape

        mean, std = self.encoder(batch.unsqueeze(1))
        if std is not None:
            var = std * std
            # real kld = sum(var - 1 - torch.log(var) + mean**2) / 2
            kld = torch.sum(
                var - 1 - torch.log(var)
                # + torch.nn.functional.relu(torch.abs(mean)-self.space)  # 0 because tanh(mean)
            )
            sample = self.decoder.parametrize(mean, std)
        else:
            kld = None
            sample = mean
        generated_batch = self.decoder(sample).squeeze(1)
        _, norm_std = self.normalization
        mse = torch.sum(torch.mean(((batch - generated_batch)/norm_std)**2, dim=(1, 2)))
        return (mse, kld)

    @property
    def normalization(self) -> tuple[float, float]:
        """Return the mean and the std of all the training data."""
        if self._normalization is None:
            warnings.warn("call `scan_data` for allowing data normalization", RuntimeWarning)
            return 0.0, 1.0
        return self._normalization

    def plot_autoencode(self, axe_input, axe_output, spots: torch.Tensor):
        """Encode and decode the images, plot the initial and regenerated images.

        Parameters
        ----------
        axe_input : matplotlib.axes.Axes
            The 2d empty axe ready to be filled by the input mosaic.
        axe_output : matplotlib.axes.Axes
            The 2d empty axe ready to be filled by the generated mosaic.
        spots : torch.Tensor
            The image stack of shape (n, h, w).
        """
        from matplotlib.axes import Axes
        assert isinstance(axe_input, Axes), axe_input.__class__.__name__
        assert isinstance(axe_output, Axes), axe_output.__class__.__name__
        assert isinstance(spots, torch.Tensor), spots.__class__.__name__
        assert spots.ndim == 3 and spots.shape[-2:] == self.shape, spots.shape

        # find the grid dimension
        height = round(math.sqrt(len(spots)*self.shape[1]/self.shape[0]))
        width = round(math.sqrt(len(spots)*self.shape[0]/self.shape[1]))
        while height * width < len(spots):
            if (height + 1) * width == len(spots):
                height += 1
            elif height * (width + 1) == len(spots):
                width += 1
            elif height < width:  # in favor of square rather than rectangle
                height += 1
            else:
                width += 1

        # input mosaic
        mosaic_in = torch.empty(
            (height*self.shape[0], width*self.shape[1]),
            dtype=spots.dtype,
            device=spots.device,
        )
        for i in range(height):
            for j in range(width):
                idx = j + i*width
                pict = spots[idx] if idx < len(spots) else 0
                mosaic_in[
                    i*self.shape[0]:(i+1)*self.shape[0], j*self.shape[1]:(j+1)*self.shape[1]
                ] = pict

        # output mosaic
        spots = self.forward(spots)
        mosaic_out = torch.empty(
            (height*self.shape[0], width*self.shape[1]),
            dtype=spots.dtype,
            device=spots.device,
        )
        for i in range(height):
            for j in range(width):
                idx = j + i*width
                pict = spots[idx] if idx < len(spots) else 0
                mosaic_out[
                    i*self.shape[0]:(i+1)*self.shape[0], j*self.shape[1]:(j+1)*self.shape[1]
                ] = pict

        # plot
        vmin = min(float(mosaic_in.min()), float(mosaic_out.min()))
        vmax = max(float(mosaic_in.max()), float(mosaic_out.max()))
        axe_input.set_title("input images")
        axe_input.axis("off")
        axe_input.imshow(
            mosaic_in.numpy(force=True),
            aspect="equal",
            interpolation=None,  # antialiasing is True
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axe_output.set_title("output images")
        axe_output.axis("off")
        axe_output.imshow(
            mosaic_out.numpy(force=True),
            aspect="equal",
            interpolation=None,  # antialiasing is True
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

    def scan_data(self, spots_generator: torch.Tensor):
        """Complete the data histogram to standardise data (centered and reduction).

        Parameters
        ----------
        spots_generator : iterable
            A generator of spot batch, each item has to be of shape (:, h, w).
        """
        assert hasattr(spots_generator, "__iter__"), spots_generator.__class__.__name__

        bins = 10000
        tot_hist = torch.zeros(bins, dtype=torch.float64)
        # for batch in tqdm(spots_generator, desc="scan data repartition"):
        for batch in spots_generator:
            assert isinstance(batch, torch.Tensor), batch.__class__.__name__
            assert batch.ndim == 3 and batch.shape[-2:] == self.shape, batch.shape
            hist, _ = torch.histogram(batch.to(torch.float64), bins=bins, range=(0.0, 1.0))
            tot_hist = tot_hist.to(hist.device)
            tot_hist += hist

        values = torch.linspace(1.0/(2*bins), 1.0 - 1.0/(2*bins), bins)
        mean = float((tot_hist * values).sum() / tot_hist.sum())
        std = float(torch.sqrt((tot_hist * (values - mean)**2).sum() / tot_hist.sum()))
        self._normalization = (mean, std)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the images."""
        return self._shape

    @property
    def space(self) -> float:
        """Return the non penalized spreading area half size."""
        return self._space


class Decoder(torch.nn.Module):
    """Decode the latent sample into a new image.

    Attributes
    ----------
    parent : laueimproc.nn.vae_spot_classifier.VAESposClassifier
        The main full auto encoder, containing this module.
    """

    def __init__(self, parent: VAESpotClassifier):
        """Initialise the decoder.

        Parameters
        ----------
        parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier
            The main module.
        """
        assert isinstance(parent, VAESpotClassifier), parent.__class__.__name__
        self._parent = (parent,)  # pack into tuple solve `cannot assign module before...`
        super().__init__()

        # size augmentation layers
        self.augmentation_layers = torch.nn.ModuleList()
        shape = parent.shape
        while min(shape) > 32:
            self.augmentation_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(12, 12, kernel_size=4, stride=2, padding=1),
                    torch.nn.Dropout(0.1),
                    torch.nn.LeakyReLU(inplace=True),
                    # torch.nn.MaxUnpool2d(2, stride=2),
                )
            )
            shape = (shape[0]//2, shape[1]//2)

        area = shape[0] * shape[1]
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(parent.latent_dim, 1000),
            # torch.nn.BatchNorm1d(1000),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.2),

            # torch.nn.Linear(500, 1000),
            # # torch.nn.BatchNorm1d(1000),
            # torch.nn.LeakyReLU(inplace=True),
            # torch.nn.Dropout(0.2),

            torch.nn.Linear(1000, 12*area),
            torch.nn.BatchNorm1d(12*area),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.2),
        )
        self.convolutive_layers = torch.nn.Sequential(
            torch.nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1),
            # torch.nn.Tanh(),
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Generate a new image from the samples.

        Parameters
        ----------
        sample : torch.Tensor
            The batch of the n samples, output of the ``Decoder.parametrize`` function.
            The size is (n, latent_dim)

        Returns
        -------
        image : torch.Tensor
            The generated image batch, of shape (n, 1, height, width)

        Notes
        -----
        No verifications are performed for performance reason.
        """
        out = sample / self.parent.space  # to have reduced data in first layer
        out = self.dense_layers(out)
        shape = 2**len(self.augmentation_layers)
        shape = (self.parent.shape[0]//shape, self.parent.shape[1]//shape)
        out = torch.reshape(out, (len(out), 12, *shape))  # not -1 for empty tensors
        for augment in self.augmentation_layers:
            out = augment(out)
        out = self.convolutive_layers(out)
        norm_mean, norm_std = self.parent.normalization
        out = out * norm_std + norm_mean  # bijection operation of normalizatiion
        return out

    @staticmethod
    def parametrize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Perform a random draw according to the normal law N(mean, std**2).

        Parameters
        ----------
        mean : torch.Tensor
            The batch of the mean vectors, shape (n, latent_dim).
        std : torch.Tensor
            The batch of the diagonal sqrt(covariance) matrix, shape (n, latent_dim).

        Returns
        -------
        draw : torch.Tensor
            The batch of the random draw.

        Notes
        -----
        No verifications are performed for performance reason.
        """
        sample = torch.randn_like(mean)  # torch.zeros_like(mean) for non variational
        sample = torch.mul(sample, std, out=(None if std.requires_grad else sample))
        sample = torch.add(
            sample, mean, out=(None if sample.requires_grad or mean.requires_grad else sample)
        )
        return sample

    @property
    def parent(self) -> VAESpotClassifier:
        """Return the parent module."""
        return self._parent[0]

    def plot_map(
        self,
        axe,
        grid: typing.Union[numbers.Integral, tuple[numbers.Integral, numbers.Integral]] = 10,
    ):
        """Generate and display spots from a regular sampling of latent space.

        Parameters
        ----------
        axe : matplotlib.axes.Axes
            The 2d empty axe ready to be filled.
        grid : int or tuple[int, int]
            Grid dimension in latent space. If only one number is supplied,
            the grid will have this dimension on all axes.
            The 2 coordinates corresponds respectively to the number of lines and columns.
        """
        from matplotlib.axes import Axes

        assert self.parent.latent_dim == 2, "can only plot in a 2d space"
        assert isinstance(axe, Axes), axe.__class__.__name__
        if isinstance(grid, numbers.Integral):
            grid = (grid, grid)
        assert isinstance(grid, tuple), grid.__class__.__name__
        assert len(grid) == 2, len(grid)
        assert isinstance(grid[0], numbers.Integral) and isinstance(grid[1], numbers.Integral), grid
        assert grid >= (1, 1), grid

        # fill figure metadata
        axe.set_title(f"generated spots from latent regular grid of {grid[0]}x{grid[1]}")

        # generated data
        device = next(self.parameters()).device
        space = self.parent.space
        points = torch.meshgrid(
            torch.linspace(-space, space, grid[0], dtype=torch.float32, device=device),
            torch.linspace(-space, space, grid[1], dtype=torch.float32, device=device),
            indexing="ij",
        )
        points = (points[0].ravel(), points[1].ravel())
        points = torch.cat([points[0].unsqueeze(1), points[1].unsqueeze(1)], dim=1)
        predicted = self.forward(points)

        # draw data
        shape = self.parent.shape
        mosaic = torch.empty(
            (grid[0]*shape[0], grid[1]*shape[1]),
            dtype=torch.float32,
            device=device,
        )
        for i in range(grid[0]):
            for j in range(grid[1]):
                mosaic[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = (
                    predicted[i*grid[1]+j]
                )
        axe.imshow(
            mosaic.numpy(force=True).transpose(),
            aspect=mosaic.shape[0]/mosaic.shape[1],
            interpolation=None,  # antialiasing is True
            cmap="gray",
            origin="lower",
            extent=[-space, space, -space, space],
        )


class Encoder(torch.nn.Module):
    """Encode an image into a gausian probality density.

    Attributes
    ----------
    parent : laueimproc.nn.vae_spot_classifier.VAESposClassifier
        The main full auto encoder, containing this module.
    """

    def __init__(self, parent: VAESpotClassifier):
        """Initialise the encoder.

        Parameters
        ----------
        parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier
            The main module.
        """
        assert isinstance(parent, VAESpotClassifier), parent.__class__.__name__
        self._parent = (parent,)  # pack into tuple solve `cannot assign module before...`
        super().__init__()

        # size reduction layers
        self.reduction_layers = torch.nn.ModuleList()
        shape = parent.shape
        while min(shape) > 32:
            self.reduction_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
                    torch.nn.Dropout(0.1),
                    torch.nn.MaxPool2d(2, stride=2),
                    torch.nn.LeakyReLU(inplace=True),
                )
            )
            shape = (shape[0]//2, shape[1]//2)

        # little fitures extraction layer
        self.convolutive_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.1),
        )

        # final dense layers
        area = shape[0] * shape[1]
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(12*area, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.2),

            # torch.nn.Linear(1000, 500),
            # # torch.nn.BatchNorm1d(500),
            # torch.nn.LeakyReLU(inplace=True),
            # torch.nn.Dropout(0.2),

        )
        self.prob_layers = torch.nn.ModuleList([
            torch.nn.Sequential(  # mean layer
                torch.nn.Linear(1000, parent.latent_dim),
                torch.nn.Tanh(),
            ),
            torch.nn.Sequential(  # log var layer
                torch.nn.Linear(1000, parent.latent_dim),
                torch.nn.LogSigmoid(),
            ),
        ])

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, typing.Union[None, torch.Tensor]]:
        """Extract the mean and the std for each images.

        Parameters
        ----------
        batch : torch.Tensor
            The stack of the n images, of shape (n, 1, height, width).

        Returns
        -------
        mean : torch.Tensor
            The mean (center of gaussians) for each image, shape (n, latent_dims).
        std : torch.Tensor
            The standard deviation (shape of gaussian) for each image, shape (n, latent_dims).

        Notes
        -----
        No verifications are performed for performance reason.
        If the model is in eval mode, it computes only the mean and gives the value None to the std.
        """
        norm_mean, norm_std = self.parent.normalization
        batch = (batch - norm_mean) / norm_std  # centered and reduced

        # prefitures extraction
        inter_value = self.convolutive_layers(batch)

        # reduction
        for reducer in self.reduction_layers:
            inter_value = reducer(inter_value)

        # final layer
        inter_value = self.dense_layers(inter_value)
        mean = self.prob_layers[0](inter_value)
        mean = mean * self.parent.space  # to have reduced data in last layer
        if not self.training:
            return (mean, None)
        std = self.prob_layers[1](inter_value)  # not std but log(std**2)
        std = torch.exp(0.5 * std)  # not std but log(std)
        return (mean, std)

    @property
    def parent(self) -> VAESpotClassifier:
        """Return the parent module."""
        return self._parent[0]

    def plot_latent(self, axe, spots_generator):
        """Plot the 2d pca of the spots projected in the latent space.

        Parameters
        ----------
        axe : matplotlib.axes.Axes
            The 2d empty axe ready to be filled.
        spots_generator : iterable
            A generator of spot batch, each item has to be of shape (:, h, w).
        """
        from matplotlib.axes import Axes
        assert isinstance(axe, Axes), axe.__class__.__name__
        assert hasattr(spots_generator, "__iter__"), spots_generator.__class__.__name__

        # eval model to get all points
        points = []
        for batch in spots_generator:
            assert isinstance(batch, torch.Tensor), batch.__class__.__name__
            assert batch.ndim == 3 and batch.shape[-2:] == self.parent.shape, batch.shape
            points.append(self.forward(batch.unsqueeze(1))[0])
        points = torch.cat(points)

        # projection with PCA
        points = points[:, :2]

        # plot
        axe.set_title("latent space projections")
        axe.axis("equal")
        space = self.parent.space
        axe.plot(
            [-space, space, space, -space, -space],
            [-space, -space, space, space, -space],
            color="black",
        )
        axe.scatter(*points.numpy(force=True).transpose())
