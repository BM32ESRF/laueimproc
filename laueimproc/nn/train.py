#!/usr/bin/env python3

"""Training pipeline for the models."""

import psutil
from torch.utils.data import DataLoader
import torch
import tqdm

from laueimproc.classes.dataset import DiagramsDataset
from laueimproc.nn.loader import SpotDataset
from laueimproc.nn.vae_spot_classifier import VAESpotClassifier


NCPU = len(psutil.Process().cpu_affinity())


def _plot_training(fig, model: VAESpotClassifier, loader: DataLoader, history: list[list[float]]):
    """Display the training statistics."""
    # plot loss
    axe_loss = fig.add_subplot(1, 3, 1)
    axe_loss.set_title("loss evolution")
    axe_loss.set_xlabel("epoch")
    axe_loss.set_ylabel("loss")
    axe_loss.set_yscale("log")
    axe_loss.plot([m for m, _ in history], label="mse")
    axe_loss.plot([k for _, k in history], label="kld")
    axe_loss.legend()

    # plot clusters
    axe_clus = fig.add_subplot(2, 3, 5)
    model.encoder.plot_latent(axe_clus, loader)

    # plot generated mosaics
    axe_map = fig.add_subplot(2, 3, 6)
    model.decoder.plot_map(axe_map)

    # plot autoencoder images
    axe_input = fig.add_subplot(2, 3, 2)
    axe_output = fig.add_subplot(2, 3, 3)
    spots = torch.cat([
        loader.dataset[i].unsqueeze(0)
        for i in range(0, len(loader.dataset), max(1, len(loader.dataset)//100-1))
    ])[:100]
    model.plot_autoencode(axe_input, axe_output, spots)


def train_vae_spot_classifier(model: VAESpotClassifier, dataset: DiagramsDataset, **kwargs):
    """Train the model.

    Parameters
    ----------
    model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier
        The initialised but not already trained model.
    dataset : laueimproc.classes.dataset.DiagramsDataset
        Contains all the initialised diagrams.
    epoch : int, default=100
        The number of epoch.
    lr : float, optional
        The learning rate.
    fig : matplotlib.figure.Figure, optional
        An empty figure ready to be filled.
    """
    assert isinstance(model, VAESpotClassifier), model.__class__.__name__
    assert isinstance(dataset, DiagramsDataset), dataset.__class__.__name__

    # data preparation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    spots_dataset = SpotDataset(dataset, model)
    batch_size = max(  # batch memory <= 10Mio
        1, 10 * 2**20 // (spots_dataset[0].dtype.itemsize * model.shape[0] * model.shape[1])
    )
    spots_loader = DataLoader(
        spots_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NCPU,
        **({"pin_memory": True, "pin_memory_device": str(device)} if device.type != "cpu" else {}),
    )
    if batch_size >= len(spots_dataset):
        spots_loader = (next(iter(spots_loader)),)

    # train
    model.train()
    optimizer = torch.optim.RAdam(
        model.parameters(),
        **{p: kwargs[p] for p in ("lr",) if p in kwargs},
    )
    epoch_log = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    history = []
    for _ in tqdm.tqdm(range(kwargs.get("epoch", 100)), unit="epoch", desc="train"):
        optimizer.zero_grad()
        history.append([0, 0])
        for spots in spots_loader:
            mse, kld = model.loss(spots.to(model.device))
            ((0.98 * mse + 0.02 * kld) / len(spots_dataset)).backward()  # add grad to leaves
            history[-1][0] += float(mse) / len(spots_dataset)
            history[-1][1] += float(kld) / len(spots_dataset)
        epoch_log.set_description_str(f"Loss: mse={history[-1][0]:.4e}, kld={history[-1][1]:.4e}")
        optimizer.step()

    model = model.to("cpu")

    # previsualisation
    if "fig" in kwargs:
        from matplotlib.figure import Figure
        assert isinstance(kwargs["fig"], Figure)
        model.eval()
        _plot_training(kwargs["fig"], model, spots_loader, history)
