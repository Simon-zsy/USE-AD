"""DeepSAD: Deep Semi-Supervised Anomaly Detection models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBNLeakyReLU(nn.Module):
    """Linear layer followed by BatchNorm1d and LeakyReLU activation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=1e-04, affine=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.linear(x)))


class DeepSADEncoder(nn.Module):
    """
    MLP encoder for DeepSAD.

    Maps input to a low-dimensional representation space.
    Uses BatchNorm and LeakyReLU activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        rep_dim: int = 32,
        bias: bool = False,
    ):
        """
        Initialize DeepSAD encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            rep_dim: Representation (output) dimension
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.rep_dim = rep_dim

        neurons = [input_dim, *hidden_dims]
        layers = [
            LinearBNLeakyReLU(neurons[i - 1], neurons[i], bias=bias)
            for i in range(1, len(neurons))
        ]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(hidden_dims[-1], rep_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class DeepSADDecoder(nn.Module):
    """MLP decoder for DeepSAD autoencoder pretraining."""

    def __init__(
        self,
        output_dim: int,
        hidden_dims: list[int] = [64, 128],
        rep_dim: int = 32,
        bias: bool = False,
    ):
        """
        Initialize DeepSAD decoder.

        Args:
            output_dim: Output (reconstruction) dimension
            hidden_dims: Hidden layer dimensions (reversed from encoder)
            rep_dim: Representation (input) dimension
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.rep_dim = rep_dim

        neurons = [rep_dim, *hidden_dims]
        layers = [
            LinearBNLeakyReLU(neurons[i - 1], neurons[i], bias=bias)
            for i in range(1, len(neurons))
        ]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(hidden_dims[-1], output_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class DeepSADAutoencoder(nn.Module):
    """Autoencoder for DeepSAD pretraining."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        rep_dim: int = 32,
        bias: bool = False,
    ):
        """
        Initialize DeepSAD autoencoder.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions for encoder
            rep_dim: Representation dimension
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = DeepSADEncoder(input_dim, hidden_dims, rep_dim, bias)
        self.decoder = DeepSADDecoder(input_dim, list(reversed(hidden_dims)), rep_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeepSADLoss(nn.Module):
    """
    Deep SAD loss function.

    For normal samples (label=0): minimize distance to hypersphere center
    For anomaly samples (label=1): maximize distance (inverse loss)
    """

    def __init__(self, eta: float = 1.0, eps: float = 1e-6):
        """
        Initialize DeepSAD loss.

        Args:
            eta: Hyperparameter controlling the margin for anomalies
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eta = eta
        self.eps = eps

    def forward(
        self,
        outputs: torch.Tensor,
        center: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DeepSAD loss.

        Args:
            outputs: Encoder outputs (batch_size, rep_dim)
            center: Hypersphere center (rep_dim,)
            labels: Sample labels (batch_size,) - 0=normal, 1=anomaly

        Returns:
            Loss value
        """
        # Distance to center
        dist = torch.sum((outputs - center) ** 2, dim=1)

        # Convert label 1 to -1 for inverse loss
        semi_targets = labels.clone()
        semi_targets[semi_targets == 1] = -1

        # Loss: normal samples minimize distance, anomalies maximize (inverse)
        losses = torch.where(
            semi_targets == 0,
            dist,
            self.eta * ((dist + self.eps) ** semi_targets.float()),
        )

        return torch.mean(losses)
