"""Deviation Network that builds on AutoEncoder."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import AutoEncoder


class DeviationNetwork(nn.Module):
    """
    Deviation Network that builds on AutoEncoder.

    Extracts three factors from AutoEncoder:
    1. Hidden representation (h): Encoder output (64-dim)
    2. Reconstruction residual vector (r): Normalized(input - reconstruction)
    3. Reconstruction error (e): L2 norm of residual (scalar)

    Anomaly score architecture:
        [r, h, e] → Dense(256, relu) + [e] → Dense(32, relu) + [e] → Dense(1, linear)

    Critical: Reconstruction error (e) is concatenated at EVERY layer.

    Args:
        input_dim: Number of input features
        autoencoder: Pre-trained AutoEncoder (optional)
    """

    def __init__(self, input_dim: int, autoencoder: Optional[AutoEncoder] = None):
        super().__init__()
        self.input_dim = input_dim

        # Build or load AutoEncoder
        if autoencoder is None:
            self.autoencoder = AutoEncoder(input_dim)
        else:
            self.autoencoder = autoencoder

        # Allow fine-tuning of AutoEncoder weights
        for param in self.autoencoder.parameters():
            param.requires_grad = True

        # Anomaly score generator
        # Input: [r (input_dim), h (64), e (1)] = input_dim + 65
        self.score_layer1 = nn.Linear(input_dim + 64 + 1, 256)
        # After layer1: [256] + [e (1)] = 257
        self.score_layer2 = nn.Linear(256 + 1, 32)
        # After layer2: [32] + [e (1)] = 33
        self.score_layer3 = nn.Linear(32 + 1, 1)

        # Initialize score generator weights
        self._init_weights()

    def _init_weights(self):
        """Xavier normal initialization for score layers"""
        for module in [self.score_layer1, self.score_layer2, self.score_layer3]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through deviation network.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            score: Anomaly score (batch_size, 1)
            reconstruction_error: L2 norm of residual (batch_size,)
            residual_normalized: Normalized residual vector (batch_size, input_dim)
        """
        # Get reconstruction and hidden representation from AutoEncoder
        reconstruction, h = self.autoencoder(x)

        # Calculate reconstruction residual (r = x - reconstruction)
        residual = x - reconstruction

        # Calculate reconstruction error (e = ||r||_2)
        reconstruction_error = torch.norm(
            residual, p=2, dim=1, keepdim=True
        )  # (batch_size, 1)

        # Normalize residual vector (r_normalized = r / ||r||_2)
        # Add epsilon to avoid division by zero
        residual_normalized = residual / (reconstruction_error + 1e-8)

        # Concatenate features: [r_normalized, h, e]
        features = torch.cat([residual_normalized, h, reconstruction_error], dim=1)

        # Anomaly score generator with e concatenated at each layer
        x = F.relu(self.score_layer1(features))
        x = torch.cat([x, reconstruction_error], dim=1)  # Concatenate e

        x = F.relu(self.score_layer2(x))
        x = torch.cat([x, reconstruction_error], dim=1)  # Concatenate e

        score = self.score_layer3(x)  # Linear activation (no activation)

        return score, reconstruction_error.squeeze(1), residual_normalized

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode: return only anomaly scores"""
        score, _, _ = self.forward(x)
        return score
