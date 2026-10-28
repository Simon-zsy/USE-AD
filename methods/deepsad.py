"""DeepSAD: Deep Semi-Supervised Anomaly Detection."""

import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from .base import BaseMethod
from model import DeepSADEncoder, DeepSADAutoencoder, DeepSADLoss


class DeepSADConfig(BaseModel):
    """Configuration for DeepSAD method."""

    pretrain_epochs: int = Field(default=100, description="Autoencoder pretraining epochs")
    epochs: int = Field(default=50, description="DeepSAD training epochs")
    batch_size: int = Field(default=128, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=1e-6, description="L2 regularization weight")
    hidden_dims: list[int] = Field(
        default=[128, 64], description="Hidden layer dimensions"
    )
    rep_dim: int = Field(default=32, description="Representation dimension")
    eta: float = Field(default=1.0, description="DeepSAD hyperparameter for anomaly loss")
    pretrain: bool = Field(default=True, description="Whether to pretrain with autoencoder")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (cuda or cpu)",
    )


class DeepSADMethod(BaseMethod):
    """
    DeepSAD: Deep Semi-Supervised Anomaly Detection.

    Two-stage training:
    1. Autoencoder pretraining (optional)
    2. DeepSAD training with hypersphere objective

    The method learns to map normal samples close to a hypersphere center
    while pushing anomalies away from the center.

    Reference: Ruff et al., "Deep Semi-Supervised Anomaly Detection" (ICLR 2020)
    """

    def __init__(self, config: DeepSADConfig):
        """
        Initialize DeepSAD method.

        Args:
            config: DeepSADConfig instance with hyperparameters
        """
        self.config = config
        self.pretrain_epochs = config.pretrain_epochs
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.hidden_dims = config.hidden_dims
        self.rep_dim = config.rep_dim
        self.eta = config.eta
        self.pretrain = config.pretrain
        self.device = config.device

    @property
    def name(self) -> str:
        """Method name."""
        return "deepsad"

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run DeepSAD method.

        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,) - 0=normal, 1=anomaly
            X_test: Test features (n_test, n_features)
            y_test: Test labels (n_test,) - 0=normal, 1=anomaly
            run_idx: Run index (for random seed variation)

        Returns:
            Dictionary with auc_roc, auc_pr, runtime_seconds
        """
        start_time = time.time()
        input_dim = X_train.shape[1]

        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]

        self.log(
            f"Training DeepSAD on {len(X_train)} samples "
            f"({len(inlier_indices)} normal, {len(outlier_indices)} anomalies)"
        )

        # Normalize data to [0, 1] for sigmoid output in autoencoder
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)

        # Create encoder
        encoder = DeepSADEncoder(
            input_dim, self.hidden_dims, self.rep_dim, bias=False
        ).to(self.device)

        # Stage 1: Autoencoder pretraining (optional)
        if self.pretrain:
            self.log(f"Stage 1: Pretraining autoencoder ({self.pretrain_epochs} epochs)...")
            autoencoder = DeepSADAutoencoder(
                input_dim, self.hidden_dims, self.rep_dim, bias=False
            ).to(self.device)

            ae_optimizer = optim.Adam(
                autoencoder.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            ae_criterion = nn.MSELoss()

            # Create data loader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            autoencoder.train()
            for epoch in range(self.pretrain_epochs):
                epoch_losses = []
                for (batch_x,) in train_loader:
                    reconstruction = autoencoder(batch_x)
                    loss = ae_criterion(reconstruction, batch_x)

                    ae_optimizer.zero_grad()
                    loss.backward()
                    ae_optimizer.step()

                    epoch_losses.append(loss.item())

                if (epoch + 1) % 20 == 0:
                    self.log(
                        f"  Pretrain Epoch [{epoch + 1}/{self.pretrain_epochs}] "
                        f"Loss: {np.mean(epoch_losses):.6f}"
                    )

            # Transfer encoder weights
            encoder.load_state_dict(autoencoder.encoder.state_dict())
            self.log("Stage 1 completed - encoder weights initialized")

        # Stage 2: DeepSAD training
        self.log(f"Stage 2: Training DeepSAD ({self.epochs} epochs)...")

        # Initialize hypersphere center
        center = self._init_center(encoder, X_train_tensor)
        self.log(f"Hypersphere center initialized (dim={self.rep_dim})")

        optimizer = optim.Adam(
            encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = DeepSADLoss(eta=self.eta).to(self.device)

        # Create data loader with labels
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        encoder.train()
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch_x, batch_y in train_loader:
                outputs = encoder(batch_x)
                loss = criterion(outputs, center, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.log(
                    f"  Epoch [{epoch + 1}/{self.epochs}] Loss: {np.mean(epoch_losses):.6f}"
                )

        self.log("Stage 2 completed")

        # Evaluation
        self.log("Evaluating on test set...")
        encoder.eval()
        with torch.no_grad():
            outputs = encoder(X_test_tensor)
            # Anomaly score = distance to center
            scores = torch.sum((outputs - center) ** 2, dim=1).cpu().numpy()

        auc_roc = roc_auc_score(y_test, scores)
        auc_pr = average_precision_score(y_test, scores)
        runtime_seconds = time.time() - start_time

        self.log(
            f"Results: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, "
            f"Runtime={runtime_seconds:.2f}s"
        )

        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "runtime_seconds": runtime_seconds,
        }

    def _init_center(
        self,
        encoder: nn.Module,
        X_train_tensor: torch.Tensor,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """
        Initialize hypersphere center as mean of encoder outputs.

        Args:
            encoder: Encoder network
            X_train_tensor: Training data tensor
            eps: Small constant to avoid zero units

        Returns:
            Hypersphere center tensor
        """
        encoder.eval()
        with torch.no_grad():
            outputs = encoder(X_train_tensor)
            center = torch.mean(outputs, dim=0)

            # Avoid zero units (can be trivially matched with zero weights)
            center[(torch.abs(center) < eps) & (center < 0)] = -eps
            center[(torch.abs(center) < eps) & (center > 0)] = eps

        return center
