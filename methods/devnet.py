"""DevNet: Deep Anomaly Detection with Deviation Networks (KDD'19)."""

import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from pydantic import BaseModel, Field

from .base import BaseMethod
from model import DevNetZScoreLoss


class DevNetConfig(BaseModel):
    """Configuration for DevNet method."""

    epochs: int = Field(default=50, description="Training epochs")
    batch_size: int = Field(default=512, description="Batch size")
    nb_batch: int = Field(default=20, description="Batches per epoch")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    hidden_dims: list[int] = Field(
        default=[128, 64], description="Hidden layer dimensions"
    )
    confidence_margin: float = Field(default=5.0, description="Deviation loss margin")
    l2_reg: float = Field(default=0.01, description="L2 regularization weight")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping norm")
    ref_size: int = Field(
        default=5000, description="Reference distribution size for z-score"
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (cuda or cpu)",
    )


class DevNetMLP(nn.Module):
    """
    DevNet MLP: Similar to SimpleMLP but outputs deviation scores (no sigmoid).

    Architecture: input → hidden_dims[0] (ReLU) → ... → 1 (Linear, no activation)
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Hidden layers (same as SimpleMLP)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: no activation (outputs deviation score in R)
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DevNetMethod(BaseMethod):
    """
    DevNet: Deep Anomaly Detection with Deviation Networks.

    Single-stage training with z-score deviation loss and 50/50 batch sampling.
    Reference: KDD'19 paper by Pang et al.
    """

    def __init__(self, config: DevNetConfig):
        """
        Initialize DevNet method.

        Args:
            config: DevNetConfig instance with hyperparameters
        """
        self.config = config
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.nb_batch = config.nb_batch
        self.learning_rate = config.learning_rate
        self.hidden_dims = config.hidden_dims
        self.confidence_margin = config.confidence_margin
        self.l2_reg = config.l2_reg
        self.gradient_clip = config.gradient_clip
        self.ref_size = config.ref_size
        self.device = config.device

    @property
    def name(self) -> str:
        return "devnet"

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run DevNet method with single-stage training.

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

        self.log(
            f"Training DevNet on {len(X_train)} samples "
            f"({len(inlier_indices)} normal, {len(outlier_indices)} anomalies)"
        )

        rng = np.random.RandomState(42 + run_idx)

        # Create model
        model = DevNetMLP(input_dim, self.hidden_dims).to(self.device)

        # RMSprop optimizer with L2 regularization (weight_decay)
        optimizer = optim.RMSprop(
            model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )

        # Z-score deviation loss
        criterion = DevNetZScoreLoss(
            confidence_margin=self.confidence_margin, ref_size=self.ref_size
        ).to(self.device)

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            epoch_losses = []

            for _ in range(self.nb_batch):
                # 50/50 batch sampling (reuse FEAWAD's strategy)
                batch_data, batch_labels = self._sample_devnet_batch(
                    X_train, outlier_indices, inlier_indices, self.batch_size, rng
                )

                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                label_tensor = torch.FloatTensor(batch_labels).to(self.device)

                # Forward
                scores = model(batch_tensor)
                loss = criterion(scores, label_tensor)

                # Backward with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.log(
                    f"  Epoch [{epoch+1}/{self.epochs}] Loss: {np.mean(epoch_losses):.6f}"
                )

        self.log("Training completed")

        # Evaluation
        self.log("Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(self.device)
            scores = model(test_tensor).cpu().numpy().flatten()

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
        (Reused from FEAWAD)

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
