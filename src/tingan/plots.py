"""tingan's plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.time import Time

import tingan.datasets


def plot_timing_noise(
    d: tingan.datasets.RealTimingNoise | tingan.datasets.TimingNoise | None,
    tin: np.ndarray | torch.Tensor,
    labels: list | np.ndarray | None = None,
    fig: plt.Figure = None,
    ax: np.ndarray = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot several examples of timing noise.

    :param d: dataset to plot.
    :param tin: a set of timing noise examples.
    :param labels: labels (e.g. probability of being real) to the examples.
    :param fig: figure to update.
    :param ax: flattened array of axes to update.

    :return: tuple of figure and axes.
    """
    if fig is None:
        fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(10, 10))
        ax = ax.flatten()
    elif ax is None:
        no_ax_error_msg = "No axes provided."
        raise ValueError(no_ax_error_msg)
    elif fig is None:
        fig = ax[0].get_figure()
    for i in range(15):
        if tin[i].ndim == 2 and d is not None:
            if d.ic:
                ax[i].plot(np.cumsum(tin[i, 0] * d.mjds_std + d.mjds_mean), tin[i, 1])
                ax[i].secondary_xaxis(
                    "top",
                    (
                        lambda x: (x - d.mjds_mean) / d.mjds_std,
                        lambda x: x * d.mjds_std + d.mjds_mean,
                    ),
                )
            else:
                ax[i].plot(tin[i, 0] * d.mjds_std + d.mjds_mean, tin[i, 1])
                ax[i].secondary_xaxis(
                    "top",
                    (
                        lambda x: (x - d.mjds_mean) / d.mjds_std,
                        lambda x: x * d.mjds_std + d.mjds_mean,
                    ),
                )
        else:
            ax[i].plot(tin[i], label=labels[i] if labels is not None else None)
            ax[i].legend()
    fig.tight_layout()
    return fig, ax


def plot_timing_noise_properties(noises: list | tuple) -> plt.Figure:
    """
    Plot properties of real and/or fake timing noise.

    Useful to assess the realism of the Generator.

    :param noises: a set of timing noises to compare.
    """
    nnoise = len(noises)

    fig = plt.figure(figsize=(10, 10))

    for n in range(nnoise):
        ax = fig.add_subplot(nnoise, 2, 2 * n + 1)
        ax.hist(
            np.mean(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="mean",
        )
        ax.hist(
            np.std(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="std",
        )
        ax.legend()

        ax = fig.add_subplot(nnoise, 2, 2 * n + 2)
        ax.scatter(np.mean(noises[n], axis=1), np.std(noises[n], axis=1))
        ax.set_xlabel("mean")
        ax.set_ylabel("std")

    fig.tight_layout()
    return fig


def plot_losses(
    generator_loss: list | np.ndarray | torch.Tensor,
    discriminator_loss: list | np.ndarray | torch.Tensor,
) -> plt.Figure:
    """
    Plot generator and discriminator losses.

    The discriminator can be trained more often than the generator, and they can have a
    different number of iterations. The first/second plot shows the
    generator/discriminatorloss as a function of generator/discriminator iterations.
    The last plot shows bth lossesas a function of generator iterations.

    :param generator_loss: the generator loss.
    :param discriminator_loss: the discriminator loss.
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax = ax.flatten()

    ig_epochs = np.linspace(0, len(generator_loss) - 1, len(generator_loss))[
        np.isnan(generator_loss)
    ]
    id_epochs = np.linspace(0, len(discriminator_loss) - 1, len(discriminator_loss))[
        np.isnan(discriminator_loss)
    ]
    if len(ig_epochs) != len(id_epochs):
        len_err_msg = (
            f"Were the generator and the discriminator train adversarially? "
            f"They seem to have been trained for a different number of "
            f"epochs: {len(ig_epochs)} and {len(id_epochs)}."
        )
        raise ValueError(len_err_msg)
    ig_epochs = np.concatenate(([0], ig_epochs)).astype(int)
    id_epochs = np.concatenate(([0], id_epochs + 1)).astype(int)

    d_iterations = [
        np.linspace(ig_epochs[ie - 1], ig_epochs[ie], id_epochs[ie] - id_epochs[ie - 1])
        for ie in range(1, len(ig_epochs))
    ]
    d_iterations = np.concatenate(d_iterations)

    ax[0].plot(generator_loss)
    ax[2].plot(generator_loss, label="Generator loss")

    if discriminator_loss is not None:
        ax[2].plot(
            d_iterations,
            discriminator_loss,
            label="Discriminator loss",
        )
        ax[1].plot(discriminator_loss, color="tab:orange")
        for ie in id_epochs[1:] - 1:
            ax[1].axvline(x=ie, color="red", linestyle="--")

    for ie in ig_epochs[1:]:
        ax[0].axvline(x=ie, color="red", linestyle="--")
        ax[2].axvline(
            x=ie,
            color="red",
            linestyle="--",
            label="End of epoch" if ie == ig_epochs[1] else None,
        )

    ax[2].legend()

    ax[0].set_xlabel("Generator iteration")
    ax[2].set_xlabel("Generator iteration")
    ax[1].set_xlabel("Discriminator iteration")

    for i in range(3):
        ax[i].set_ylabel("Loss")

    fig.tight_layout()
    return fig


def plot_labels(
    labels_real: list | np.ndarray, labels_fake: list | np.ndarray
) -> plt.Figure:
    """
    Plot real and fake data numerical labels.

    Labels are outputs from the generator, i.e., the probability that the data is real.
    Ideally, labels_real (fake_labels) should be close to 1 (0) during the first
    iterations and close to 0.5 (0.5) at the end for training.

    :param labels_real: labels for real data.
    :param labels_fake: labels for fake data.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels_real, label="Real")
    ax.plot(labels_fake, label="Fake")
    i_epochs = np.linspace(0, len(labels_real) - 1, len(labels_real))[
        np.isnan(labels_real)
    ]
    for ie in i_epochs:
        ax.axvline(
            x=ie,
            color="red",
            linestyle="--",
            label="End of epoch" if ie == i_epochs[0] else None,
        )
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Label")
    fig.tight_layout()
    return fig


def plot_timellm_residuals(file: Path) -> plt.Figure:
    """
    Plot residuals stored in a file in timellm-compatible format.

    :param file: path to timellm file.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data = pd.read_csv(file)
    ndates = len(data["date"])
    dates = np.ones(ndates) * np.nan
    for i in range(ndates):
        dates[i] = Time(data["date"][i], format="isot").mjd
    ax.plot(dates, data["resid_s"])
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residuals [s]")
    fig.tight_layout()
    return fig
