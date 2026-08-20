"""tingan's networks."""

import torch


class Generator(torch.nn.Module):
    """
    Generator network.

    Tries to generate realistic (fake) noise.
    """

    def __init__(self, nz: int) -> None:
        """Initialize the generator."""
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 8 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, nz),
        )

    def forward(self, random_noise: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.main(random_noise)


class Discriminator(torch.nn.Module):
    """
    Discriminator network.

    Tries to discriminate between timing noise and fake noise.
    """

    def __init__(self, nz: int) -> None:
        """Initialize the discriminator."""
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 8 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nz, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        d = self.main(noise)
        if len(d.shape) == 3:
            d = torch.mean(d, dim=(1, 2))
        return d


class TimeSeriesDiscriminator(torch.nn.Module):
    """
    A 1D-CNN-based discriminator that classifies a time series segment as real or fake.

    :param seq_len: length of the input sequence (forecast horizon)
    :param n_channels: number of variables (channels) in the multivariate series
    """

    def __init__(self, seq_len: int, n_channels: int = 1) -> None:
        """Initialize the discriminator."""
        super().__init__()

        self.net = torch.nn.Sequential(
            # Input shape: (batch, n_channels, seq_len)
            torch.nn.Conv1d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, seq_len)
            flatten_dim = self.net(dummy).numel()

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flatten_dim, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),  # output probability of "real"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = x.permute(0, 2, 1)  # (batch, n_channels, seq_len) for Conv1d
        features = self.net(x)
        return self.head(features)


def trainable_parameters(model: torch.nn.Module) -> list:
    """
    Identify the trainable parameters of a neural network.

    :param model: PyTorch neural network
    :return: list of trainable parameters
    """
    return [p for p in model.parameters() if p.requires_grad]
