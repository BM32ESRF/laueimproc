#!/usr/bin/env python3

"""Training pipeline for the models."""

from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch

from laueimproc.classes.dataset import DiagramsDataset
from laueimproc.nn.loader import SpotDataloader
from laueimproc.nn.vae_spot_classifier import VAESpotClassifier


def _plot_training(fig, model: VAESpotClassifier, loader: DataLoader, history: list[list[float]]):
    """Display the training statistics."""
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure), fig.__class__.__name__

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
        loader[i].unsqueeze(0)
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
    batch : int, optional
        The number of pictures in each batch.
        By default, the batch size is equal to the dataset size,
        so that there is exactely one complete batch per epoch.
    epoch : int, default=10
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

    spots_loader = SpotDataloader(dataset, model)
    batch_size = max(  # batch memory <= 10Mio
        1, 10 * 2**20 // (torch.float32.itemsize * model.shape[0] * model.shape[1])
    )
    batch_size = min(batch_size, kwargs.get("batch", len(spots_loader)))
    spots_loader.batch_size = batch_size

    # make stat about data repartition
    model.scan_data(spots_loader)

    # train
    model.train()
    optimizer = torch.optim.RAdam(
        model.parameters(),
        **{p: kwargs[p] for p in ("lr",) if p in kwargs},
    )
    epoch_log = tqdm(total=0, position=1, bar_format="{desc}")
    history = []
    batch_size = kwargs.get("batch", len(spots_loader))
    cum_batch = 0
    optimizer.zero_grad()
    for _ in tqdm(range(kwargs.get("epoch", 10)), unit="epoch", desc="train"):
        history.append([0, 0])
        for spots in spots_loader:
            if cum_batch + len(spots) >= batch_size:
                spots_last, spots = spots[:batch_size-cum_batch], spots[batch_size-cum_batch:]
                mse, kld = model.loss(spots_last.to(model.device))
                spots_last.to("cpu")  # because it can be cached, leading to cuda leak
                ((0.9 * mse + 0.1 * kld) / len(spots_loader)).backward()  # add grad to leaves
                history[-1][0] += float(mse) / len(spots_loader)
                history[-1][1] += float(kld) / len(spots_loader)
                optimizer.step()
                optimizer.zero_grad()
            if len(spots) == 0:
                continue
            mse, kld = model.loss(spots.to(model.device))
            ((0.9 * mse + 0.1 * kld) / len(spots_loader)).backward()  # add grad to leaves
            history[-1][0] += float(mse) / len(spots_loader)
            history[-1][1] += float(kld) / len(spots_loader)
        epoch_log.set_description_str(f"Loss: mse={history[-1][0]:.4e}, kld={history[-1][1]:.4e}")

    model = model.to("cpu")

    # previsualisation
    if "fig" in kwargs:
        model.eval()
        _plot_training(kwargs["fig"], model, spots_loader, history)
