"""PReNet: Pairwise Ranking Network for anomaly detection."""

import torch
import torch.nn as nn


class PReNet(nn.Module):
    """
    Pairwise Ranking Network for anomaly detection.

    Architecture:
    - Feature extractor: input_dim -> 20 (ReLU)
    - Concatenate left and right features: 40
    - Regression layer: 40 -> 1

    Reference: "Deep Weakly-supervised Anomaly Detection"
    """

    def __init__(self, input_dim: int, hidden_dim: int = 20):
        """
        Initialize PReNet.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (default: 20)
        """
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.reg = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pairwise input.

        Args:
            x_left: Left sample features (batch_size, input_dim)
            x_right: Right sample features (batch_size, input_dim)

        Returns:
            Anomaly scores (batch_size,)
        """
        feature_left = self.feature(x_left)
        feature_right = self.feature(x_right)

        feature = torch.cat((feature_left, feature_right), dim=1)
        score = self.reg(feature)

        return score.squeeze(-1)
