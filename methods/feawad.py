"""FEAWAD: Few-shot Anomaly detection with Weak Supervision."""

import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import BaseMethod
from model import AutoEncoder, DeviationNetwork, FeawadDevLoss, FeawadDevNetLoss


class FEAWADConfig(BaseModel):
    """Configuration for FEAWAD method."""

    ae_epochs: int = Field(default=100, description="AutoEncoder training epochs")
    devnet_epochs: int = Field(
        default=30, description="DeviationNetwork training epochs"
    )
    batch_size: int = Field(default=512, description="Batch size")
    nb_batch: int = Field(default=20, description="Batches per epoch")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    confidence_margin: float = Field(default=5.0, description="Loss margin")
    simplified_loss: bool = Field(
        default=False,
        description="Use simplified loss (FeawadDevLoss) instead of z-score loss (FeawadDevNetLoss)",
    )
    ref_size: int = Field(
        default=5000,
        description="Reference distribution size for z-score (only used when simplified_loss=False)",
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (cuda or cpu)",
    )


class FEAWADMethod(BaseMethod):
    """
    FEAWAD: Few-shot Anomaly detection with Weak Supervision.

    Two-stage training:
    - Stage 1: AutoEncoder pre-training on normal samples
    - Stage 2: DeviationNetwork end-to-end training with labeled anomalies

    Loss variants:
    - simplified_loss=True: Uses FeawadDevLoss (margin-based, no z-score)
    - simplified_loss=False: Uses FeawadDevNetLoss (z-score normalization with ref_size)
    """

    def __init__(self, config: FEAWADConfig):
        """
        Initialize FEAWAD method.

        Args:
            config: FEAWADConfig instance with hyperparameters
        """
        self.config = config
        self.ae_epochs = config.ae_epochs
        self.devnet_epochs = config.devnet_epochs
        self.batch_size = config.batch_size
        self.nb_batch = config.nb_batch
        self.learning_rate = config.learning_rate
        self.confidence_margin = config.confidence_margin
        self.simplified_loss = config.simplified_loss
        self.ref_size = config.ref_size
        self.device = config.device

    @property
    def name(self) -> str:
        return "feawad"

    def _create_criterion(self) -> nn.Module:
        """Create loss criterion based on configuration."""
        if self.simplified_loss:
            return FeawadDevLoss(confidence_margin=self.confidence_margin)
        else:
            return FeawadDevNetLoss(
                confidence_margin=self.confidence_margin, ref_size=self.ref_size
            )

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run FEAWAD method with two-stage training.

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

        # Get indices
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]

        loss_type = "simplified" if self.simplified_loss else "z-score"
        self.log(
            f"Training {self.name} ({loss_type} loss) on {len(X_train)} samples "
            f"({len(inlier_indices)} normal, {len(outlier_indices)} anomalies)"
        )

        rng = np.random.RandomState(42 + run_idx)

        # ==== Stage 1: AutoEncoder Pre-training ====
        self.log(f"Stage 1: Pre-training AutoEncoder ({self.ae_epochs} epochs)...")

        autoencoder = AutoEncoder(input_dim).to(self.device)
        ae_optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        ae_criterion = nn.MSELoss()

        for epoch in range(self.ae_epochs):
            autoencoder.train()
            epoch_losses = []

            for _ in range(self.nb_batch):
                # Sample from normal samples only
                batch_indices = rng.choice(
                    inlier_indices, size=self.batch_size, replace=True
                )
                batch_tensor = torch.FloatTensor(X_train[batch_indices]).to(self.device)

                reconstruction, _ = autoencoder(batch_tensor)
                loss = ae_criterion(reconstruction, batch_tensor)

                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                self.log(
                    f"  Epoch [{epoch + 1}/{self.ae_epochs}] Loss: {np.mean(epoch_losses):.6f}"
                )

        self.log("Stage 1 completed")

        # ==== Stage 2: DeviationNetwork Training ====
        self.log(
            f"Stage 2: Training DeviationNetwork ({self.devnet_epochs} epochs)..."
        )

        devnet = DeviationNetwork(input_dim, autoencoder=autoencoder).to(self.device)
        devnet_optimizer = optim.Adam(devnet.parameters(), lr=self.learning_rate)
        devnet_criterion = self._create_criterion().to(self.device)

        for epoch in range(self.devnet_epochs):
            devnet.train()
            epoch_losses = []

            for _ in range(self.nb_batch):
                # 50/50 sampling
                batch_data, batch_labels = self._sample_devnet_batch(
                    X_train, outlier_indices, inlier_indices, self.batch_size, rng
                )

                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                label_tensor = torch.FloatTensor(batch_labels).to(self.device)

                score, reconstruction_error, _ = devnet(batch_tensor)
                loss = devnet_criterion(score, reconstruction_error, label_tensor)

                devnet_optimizer.zero_grad()
                loss.backward()
                devnet_optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 5 == 0:
                self.log(
                    f"  Epoch [{epoch + 1}/{self.devnet_epochs}] Loss: {np.mean(epoch_losses):.6f}"
                )

        self.log("Stage 2 completed")

        # ==== Evaluation ====
        self.log("Evaluating on test set...")
        devnet.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(self.device)
            scores = devnet.predict(test_tensor)

        scores_np = scores.cpu().numpy().flatten()
        auc_roc = roc_auc_score(y_test, scores_np)
        auc_pr = average_precision_score(y_test, scores_np)
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

    def _sample_devnet_batch(
        self,
        X_train: np.ndarray,
        outlier_indices: np.ndarray,
        inlier_indices: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
    ):
        """
        Sample batch with 50/50 strategy: even indices=normal, odd indices=anomaly.

        Args:
            X_train: Training features
            outlier_indices: Indices of anomalies
            inlier_indices: Indices of normal samples
            batch_size: Batch size
            rng: Random state

        Returns:
            batch_data: (batch_size, n_features)
            batch_labels: (batch_size,)
        """
        batch_data = np.empty((batch_size, X_train.shape[1]))
        batch_labels = np.empty(batch_size)

        for i in range(batch_size):
            if i % 2 == 0:  # Even → normal
                idx = rng.choice(inlier_indices, 1)[0]
                batch_data[i] = X_train[idx]
                batch_labels[i] = 0
            else:  # Odd → anomaly
                idx = rng.choice(outlier_indices, 1)[0]
                batch_data[i] = X_train[idx]
                batch_labels[i] = 1

        return batch_data, batch_labels
