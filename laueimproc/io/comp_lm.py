#!/usr/bin/env python3

"""Implement a convolutive generative variational auto-encoder neuronal network."""


import torch


class VariationalEncoder(torch.nn.Module):
    """Projects images into a more compact space.

    Each patch of 192x192 pixels with a stride of 64 pixels
    is projected into a space of dimension 64.
    """

    def __init__(self):
        super().__init__()

        eta = 1.26  # (lat_dim/first_dim)**(1/nb_layers)

        self.pre = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1),
            torch.nn.ELU(),
        )
        self.encoder = torch.nn.Sequential(
            *(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        round(16*eta**layer),
                        round(16*eta**(layer+1)),
                        kernel_size=6,
                        stride=2,
                        padding=2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(round(16*eta**(layer+1))),
                    torch.nn.ELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Conv2d(
                        round(16*eta**(layer+1)),
                        round(16*eta**(layer+1)),
                        kernel_size=3,
                        stride=1
                    ),
                    torch.nn.ELU(),
                    torch.nn.Dropout(0.1),
                )
                for layer in range(6)
            ),
        )
        self.post = torch.nn.Sequential(
            torch.nn.Conv2d(65, 64, kernel_size=3, stride=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the function on the images.

        Parameters
        ----------
        img : torch.Tensor
            The float image batch of shape (n, 3, h, w).
            With h and w >= 192 + k*32, k positive integer.

        Returns
        -------
        lat : torch.Tensor
            The projection of the image in the latent space.
            New shape is (n, 256, (h-160)/32, (w-160)/32) with value in [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import VariationalEncoder
        >>> encoder = VariationalEncoder()
        >>> encoder(torch.rand((10, 1, 192, 192+2*32))).shape
        torch.Size([10, 256, 1, 3])
        >>>
        """
        assert isinstance(img, torch.Tensor), img.__class__.__name__
        assert img.ndim == 4, img.shape
        assert img.shape[1] == 1, img.shape
        # assert img.shape[2:] >= (192, 192), img.shape
        # assert img.shape[2] % 32 == 0, img.shape
        # assert img.shape[3] % 32 == 0, img.shape
        assert img.dtype.is_floating_point, img.dtype

        mean = (
            torch.mean(img, dim=(2, 3), keepdim=True)
            .expand(-1, 1, img.shape[2]//64, img.shape[3]//64)
        )
        x = self.pre(img)
        x = self.encoder(x)
        lat = self.post(torch.cat((x, mean), dim=1))
        if self.training:
            lat = self.add_quantization_noise(lat)
        return lat

    @staticmethod
    def add_quantization_noise(lat: torch.Tensor) -> torch.Tensor:
        """Add a uniform noise in order to simulate the quantization into uint8.

        Parameters
        ----------
        lat : torch.Tensor
            The float lattent space of shape (n, 256, a, b) with value in range ]0, 1[.

        Returns
        -------
        noised_lat : torch.Tensor
            The input tensor with a aditive uniform noise U(-.5/255, .5/255).
            The finals values are clamped to stay in the range [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import VariationalEncoder
        >>> lat = torch.rand((10, 256, 1, 3))
        >>> q_lat = VariationalEncoder.add_quantization_noise(lat)
        >>> torch.all(abs(q_lat - lat) <= 0.5/255)
        tensor(True)
        >>> abs((q_lat - lat).mean().round(decimals=4))
        tensor(0.)
        >>>
        """
        assert isinstance(lat, torch.Tensor), lat.__class__.__name__
        assert lat.ndim == 4, lat.shape
        assert lat.shape[1] == 256, lat.shape
        assert lat.dtype.is_floating_point, lat.dtype

        noise = torch.rand_like(lat)/255
        noise -= 0.5/255
        out = lat + noise
        out = torch.clamp(out, min=0, max=1)
        return out


class Decoder(torch.nn.Module):
    """Unfold the projected encoded images into the color space."""

    def __init__(self):
        super().__init__()

        eta = 1.605

        self.pre = torch.nn.Sequential(
            torch.nn.ConstantPad2d(2, 0.5),
            torch.nn.Conv2d(258, 256, kernel_size=1),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            *(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        round(24*eta**layer),
                        round(24*eta**(layer-1)),
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(round(24*eta**(layer-1))),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.1),
                    torch.nn.Conv2d(
                        round(24*eta**(layer-1)),
                        round(24*eta**(layer-1)),
                        kernel_size=3,
                        stride=1,
                        padding=2,
                    ),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.1),
                )
                for layer in range(5, 0, -1)
            ),
        )
        self.post = torch.nn.Sequential(
            torch.nn.Conv2d(24, 12, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(12, 3, kernel_size=5),
            torch.nn.Sigmoid(),
        )


    def forward(self, lat: torch.Tensor) -> torch.Tensor:
        """Apply the function on the latent images.

        Parameters
        ----------
        lat : torch.Tensor
            The projected image in the latent space of shape (n, 256, hl, wl).

        Returns
        -------
        img : torch.Tensor
            A close image in colorspace to the input image.
            It is as mutch bijective as possible than VariationalEncoder.
            New shape is (n, 256, 160+hl*32, 160+wl*32) with value in [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import Decoder
        >>> decoder = Decoder()
        >>> mse, gen = decoder(torch.rand((10, 256, 1, 3)))
        >>> mse.shape
        torch.Size([10, 3, 192, 256])
        >>> gen.shape
        torch.Size([10, 3, 192, 256])
        >>>
        """
        assert isinstance(lat, torch.Tensor), lat.__class__.__name__
        assert lat.ndim == 4, lat.shape
        assert lat.shape[1] == 256, lat.shape
        assert lat.shape[2:] >= (1, 1), lat.shape
        assert lat.dtype.is_floating_point, lat.dtype

        pos_h = torch.linspace(-1, 1, lat.shape[2], dtype=lat.dtype, device=lat.device)
        pos_w = torch.linspace(-1, 1, lat.shape[3], dtype=lat.dtype, device=lat.device)
        pos_h, pos_w = pos_h.reshape(1, 1, -1, 1), pos_w.reshape(1, 1, 1, -1)
        pos_h, pos_w = (
            pos_h.expand(len(lat), 1, *lat.shape[2:]),
            pos_w.expand(len(lat), 1, *lat.shape[2:]),
        )

        x = self.pre(torch.cat((lat, pos_h, pos_w), dim=1))
        x = self.decoder(x)
        out = self.post(x[:, :, 11:-11, 11:-11])
        return out
