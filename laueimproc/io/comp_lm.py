#!/usr/bin/env python3

"""Implement a convolutive generative variational auto-encoder neuronal network."""

import lzma
import pathlib
import pickle
import typing

import numpy as np
import torch
import tqdm

from laueimproc.classes.diagram import Diagram


class VariationalEncoder(torch.nn.Module):
    """Projects images into a more compact space.

    Each patch of 320x320 pixels with a stride of 64 pixels
    is projected into a space of dimension 64, quantizable with 8 bits per component.
    """

    def __init__(self):
        super().__init__()

        eta = 1.32  # (lat_dim/first_dim)**(1/nb_layers)

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
                    torch.nn.Conv2d(
                        round(16*eta**(layer+1)),
                        round(16*eta**(layer+1)),
                        kernel_size=3,
                    ),
                    torch.nn.ELU(),
                )
                for layer in range(5)
            ),
        )
        self.post = torch.nn.Sequential(
            torch.nn.Conv2d(65, 64, kernel_size=3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the function on the images.

        Parameters
        ----------
        img : torch.Tensor
            The float image batch of shape (n, 1, h, w).
            With h and w >= 160 + k*32, k >= 0 integer.

        Returns
        -------
        lat : torch.Tensor
            The projection of the image in the latent space.
            New shape is (n, 256, h/32-4, w/32-4) with value in [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import VariationalEncoder
        >>> encoder = VariationalEncoder()
        >>> encoder(torch.rand((10, 1, 160, 256))).shape
        torch.Size([10, 64, 1, 4])
        >>>
        """
        assert isinstance(img, torch.Tensor), img.__class__.__name__
        assert img.ndim == 4, img.shape
        assert img.shape[1] == 1, img.shape
        assert img.shape[2:] >= (160, 160), img.shape
        assert img.shape[2] % 32 == 0, img.shape
        assert img.shape[3] % 32 == 0, img.shape
        assert img.dtype.is_floating_point, img.dtype

        mean = (
            torch.mean(img, dim=(2, 3), keepdim=True)
            .expand(-1, 1, img.shape[2]//32-4, img.shape[3]//32-4)
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
            The float lattent space of shape (n, 64, a, b) with value in range ]0, 1[.

        Returns
        -------
        noised_lat : torch.Tensor
            The input tensor with a aditive uniform noise U(-.5/255, .5/255).
            The finals values are clamped to stay in the range [0, 1].

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import VariationalEncoder
        >>> lat = torch.rand((10, 64, 1, 4))
        >>> q_lat = VariationalEncoder.add_quantization_noise(lat)
        >>> torch.all(abs(q_lat - lat) <= 0.5/255)
        tensor(True)
        >>> abs((q_lat - lat).mean().round(decimals=4))
        tensor(0.)
        >>>
        """
        assert isinstance(lat, torch.Tensor), lat.__class__.__name__
        assert lat.ndim == 4, lat.shape
        assert lat.shape[1] == 64, lat.shape
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

        eta = 1.32

        self.pre = torch.nn.Sequential(
            torch.nn.ConstantPad2d(2, 0.5),
            torch.nn.Conv2d(66, 64, kernel_size=1),
            torch.nn.ELU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            *(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        round(16*eta**layer),
                        round(16*eta**(layer-1)),
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(round(16*eta**(layer-1))),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Conv2d(
                        round(16*eta**(layer-1)),
                        round(16*eta**(layer-1)),
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.ELU(inplace=True),
                )
                for layer in range(5, 0, -1)
            ),
        )
        self.post = torch.nn.Sequential(
            torch.nn.Conv2d(16, 12, kernel_size=5, padding=2),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(12, 1, kernel_size=5, padding=2),
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
        >>> decoder(torch.rand((10, 64, 1, 4))).shape
        torch.Size([10, 1, 160, 256])
        >>>
        """
        assert isinstance(lat, torch.Tensor), lat.__class__.__name__
        assert lat.ndim == 4, lat.shape
        assert lat.shape[1] == 64, lat.shape
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
        out = self.post(x)
        return out


class LMCodec(torch.nn.Module):
    """Encode and Decode Laue Max images."""

    def __init__(self, weights: typing.Union[str, bytes, pathlib.Path]):
        """Initialise the codec

        Parameters
        ----------
        weights : pathlike
            The filename of the model data.
        """
        assert isinstance(weights, (str, bytes, pathlib.Path)), weights.__class__.__name__
        weights = pathlib.Path(weights).expanduser().resolve()
        assert not weights.is_dir(), weights

        super().__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()
        self._weights = weights

        if weights.is_file():
            checkpoint = torch.load(weights)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
        else:
            torch.save(
                {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()},
                weights,
            )

    def decode(
        self, data: typing.Union[tuple[torch.Tensor, tuple[int, int, int, int]], bytes]
    ) -> typing.Union[torch.Tensor, np.ndarray]:
        """Decode le compressed content.

        Parameters
        ----------
        data
            The compact data representation.

        Examples
        --------
        >>> import numpy as np
        >>> from laueimproc.io.comp_lm import LMCodec
        >>> codec = LMCodec("/tmp/lmweights.tar").eval()
        >>> img = np.random.randint(0, 65536, (2000, 2000), dtype=np.uint16)
        >>> data = codec.encode(img)
        >>> decoded = codec.decode(data)
        >>> (img == decoded).all()
        True
        >>>
        """
        if isinstance(data, tuple):
            assert len(data) == 2, len(data)
            encoded, pad = data
            assert isinstance(encoded, torch.Tensor), encoded.__class__.__name__
            assert encoded.ndim == 4, encoded.shape
            assert encoded.shape[1] == 64, encoded.shape
            assert len(pad) == 4, len(pad)
            assert all(isinstance(p, int) for p in pad), pad
            padleft, padright, padtop, padbottom = pad
            img = self.decoder(encoded)
            img = img[:, :, padtop:img.shape[2]-padbottom, padleft:img.shape[3]-padright]
            return img

        assert isinstance(data, bytes), data.__class__.__name__
        data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
        lat_np, residu, (padleft, padright, padtop, padbottom) = pickle.loads(data)
        lat_torch = torch.from_numpy(lat_np).to(torch.float32) / 255
        pred_torch = self.decoder(lat_torch)
        pred_torch = (
            pred_torch
            [:, :, padtop:pred_torch.shape[2]-padbottom, padleft:pred_torch.shape[3]-padright]
        )
        pred_np = (pred_torch*65535 + 0.5).squeeze(0).squeeze(0).numpy(force=True).astype(np.int32)
        img_np = (residu + pred_np).astype(np.uint16)
        return img_np

    def encode(
        self, img: typing.Union[torch.Tensor, np.ndarray]
    ) -> typing.Union[tuple[torch.Tensor, tuple[int, int, int, int]], bytes]:
        """Encode the image

        Parameters
        ----------
        img : torch.Tensor or np.ndarray
            The 1 channel image to encode.

        Examples
        --------
        >>> import numpy as np
        >>> from laueimproc.io.comp_lm import LMCodec
        >>> codec = LMCodec("/tmp/lmweights.tar").eval()
        >>> img = np.random.randint(0, 65536, (2000, 2000), dtype=np.uint16)
        >>> encoded = codec.encode(img)
        >>>
        """
        if isinstance(img, torch.Tensor):
            assert img.ndim == 4, img.shape
            assert img.shape[1] == 1, img.shape

            # padding
            padtop = max(0, 128-img.shape[2])
            padtop += 32 - ((img.shape[2] + padtop) % 32)
            padleft = max(0, 1280-img.shape[3])
            padleft += 32 - ((img.shape[3] + padleft) % 32)
            padtop, padbottom = padtop//2, padtop - padtop//2
            padleft, padright = padleft//2, padleft - padleft//2
            img = torch.nn.functional.pad(img, (padleft, padright, padtop, padbottom))

            # transform
            return self.encoder(img), (padleft, padright, padtop, padbottom)

        # case numpy uint16
        assert isinstance(img, np.ndarray), img.__class__.__name__
        assert img.ndim == 2, img.shape
        assert img.dtype == np.uint16
        img_torch = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 65535
        lat_torch, pad = self.encode(img_torch)
        lat_np = (lat_torch*255 + 0.5).numpy(force=True).astype(np.uint8)
        lat_torch_quant = torch.from_numpy(lat_np).to(torch.float32) / 255
        pred_torch = self.decode((lat_torch_quant, pad))
        pred_np = (pred_torch*65535 + 0.5).squeeze(0).squeeze(0).numpy(force=True).astype(np.int32)
        residu = img.astype(np.int32) - pred_np
        data = pickle.dumps((lat_np, residu, pad))
        data = lzma.compress(data, format=lzma.FORMAT_ALONE, check=lzma.CHECK_NONE, preset=9)
        return data

    def forward(self, img: typing.Union[torch.Tensor, np.ndarray, bytes]) -> torch.Tensor:
        """Encode then decode the image.

        Parameters
        ----------
        img : torch.Tensor
            The 1 channel image to encode.

        Returns
        -------
        predicted : torch.Tensor
            The predicted image of same shape.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.io.comp_lm import LMCodec
        >>> codec = LMCodec("/tmp/lmweights.tar")
        >>> codec.forward(torch.rand((2000, 2000))).shape
        torch.Size([2000, 2000])
        >>>
        """
        assert isinstance(img, torch.Tensor), img.__class__.__name__
        assert 2 <= img.ndim <= 4, img.shape
        if img.ndim < 4:
            return torch.squeeze(self.forward(torch.unsqueeze(img, 0)), 0)
        return self.decode(self.encode(img))

    def overfit(self, diagrams: list[Diagram]):
        """Overfit the model on the given diagrams.

        Parameters
        ----------
        diagrams : list[Diagram]
            All the diagrams we want to compress.

        Examples
        --------
        >>> from laueimproc import Diagram
        >>> from laueimproc.io.comp_lm import LMCodec
        >>> from laueimproc.io.download import get_samples
        >>> codec = LMCodec("/tmp/lmweights.tar")
        >>> diagrams = [Diagram(f) for f in get_samples().glob("*.jp2")]
        >>> codec.overfit(diagrams)
        >>>
        """
        self.train()
        optim = torch.optim.RAdam(self.parameters(), lr=5e-5)
        loss = torch.nn.MSELoss()

        self.to("cuda")

        for epoch in range(100):
            tot_loss_val = 0
            for i, diagram in enumerate(tqdm.tqdm(diagrams, unit="img", desc=f"epoch {epoch+1}")):
                if i % 8 == 0:
                    optim.zero_grad(set_to_none=True)
                img = diagram.image.clone().to("cuda")
                pred = self.forward(img)
                loss_val = loss(pred, img)
                tot_loss_val += loss_val.item()
                loss_val.backward()
                if (i+1) % 8 == 0:
                    optim.step()
            print(tot_loss_val/len(diagrams))
        torch.save(
            {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()},
            self._weights,
        )
        no_comp = 0
        comp = 0
        self.eval()
        self.to("cpu")
        for diagram in tqdm.tqdm(diagrams):
            img = (diagram.image*65535 + 0.5).numpy(force=True).astype(np.uint16)
            no_comp += len(img.tobytes())
            comp += len(self.encode(img))
        print(f"total no compressed {no_comp} bytes")
        print(f"total compressed {comp} bytes")
        print(f"compression factor {no_comp/comp:.2f}")
