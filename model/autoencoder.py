"""AutoEncoder for feature encoding and reconstruction."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    AutoEncoder for feature encoding and reconstruction.

    Architecture:
        Input(dim) → Dense(128, relu) → Dense(64, relu)
                  → Dense(128, relu) → Dense(dim, relu)

    Args:
        input_dim: Number of input features
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        # Encoder layers
        self.encoder1 = nn.Linear(input_dim, 128)
        self.encoder2 = nn.Linear(128, 64)

        # Decoder layers
        self.decoder1 = nn.Linear(64, 128)
        self.decoder2 = nn.Linear(128, input_dim)

        # Initialize weights with Xavier normal (Keras glorot_normal equivalent)
        self._init_weights()

    def _init_weights(self):
        """Xavier normal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            reconstruction: Reconstructed input (batch_size, input_dim)
            h: Hidden representation (batch_size, 64)
        """
        # Encoder
        x = F.relu(self.encoder1(x))
        h = F.relu(self.encoder2(x))  # Hidden representation

        # Decoder
        x = F.relu(self.decoder1(h))
        reconstruction = F.relu(self.decoder2(x))

        return reconstruction, h
