"""PReNet: Deep Weakly-supervised Anomaly Detection with Pairwise Ranking."""

import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import BaseMethod
from model import PReNet


class PReNetConfig(BaseModel):
    """Configuration for PReNet method."""

    epochs: int = Field(default=50, description="Training epochs")
    batch_size: int = Field(default=512, description="Batch size")
    nb_batch: int = Field(default=20, description="Batches per epoch")
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    weight_decay: float = Field(default=1e-2, description="L2 regularization weight")
    hidden_dim: int = Field(default=20, description="Hidden layer dimension")
    s_a_a: float = Field(default=8.0, description="Target score for (anomaly, anomaly) pairs")
    s_a_u: float = Field(default=4.0, description="Target score for (anomaly, unlabeled) pairs")
    s_u_u: float = Field(default=0.0, description="Target score for (unlabeled, unlabeled) pairs")
    num_ref: int = Field(default=30, description="Number of reference samples for scoring")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (cuda or cpu)",
    )


class PReNetMethod(BaseMethod):
    """
    PReNet: Deep Weakly-supervised Anomaly Detection.

    Uses pairwise ranking to learn anomaly scores by comparing pairs of samples.
    Training uses stratified pair sampling with three pair types:
    - (a,a): anomaly-anomaly pairs
    - (a,u): anomaly-unlabeled pairs
    - (u,u): unlabeled-unlabeled pairs

    Reference: "Deep Weakly-supervised Anomaly Detection"
    """

    def __init__(self, config: PReNetConfig):
        """
        Initialize PReNet method.

        Args:
            config: PReNetConfig instance with hyperparameters
        """
        self.config = config
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.nb_batch = config.nb_batch
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.hidden_dim = config.hidden_dim
        self.s_a_a = config.s_a_a
        self.s_a_u = config.s_a_u
        self.s_u_u = config.s_u_u
        self.num_ref = config.num_ref
        self.device = config.device

    @property
    def name(self) -> str:
        """Method name."""
        return "prenet"

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run PReNet method.

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
            f"Training PReNet on {len(X_train)} samples "
            f"({len(inlier_indices)} normal, {len(outlier_indices)} anomalies)"
        )

        rng = np.random.RandomState(42 + run_idx)

        # Store training data for inference
        self.X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        self.y_train = y_train

        # Create model
        model = PReNet(input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.RMSprop(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            epoch_losses = []

            for _ in range(self.nb_batch):
                # Generate stratified pair batch
                X_left, X_right, y_batch = self._sample_pairs(
                    X_train, outlier_indices, inlier_indices, self.batch_size, rng
                )

                X_left = torch.FloatTensor(X_left).to(self.device)
                X_right = torch.FloatTensor(X_right).to(self.device)
                y_batch = torch.FloatTensor(y_batch).to(self.device)

                # Forward
                scores = model(X_left, X_right)

                # MAE loss
                loss = torch.mean(torch.abs(y_batch - scores))

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.log(
                    f"  Epoch [{epoch + 1}/{self.epochs}] Loss: {np.mean(epoch_losses):.6f}"
                )

        self.log("Training completed")

        # Evaluation
        self.log("Evaluating on test set...")
        model.eval()
        scores = self._predict_scores(model, X_test, rng)

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

    def _sample_pairs(
        self,
        X_train: np.ndarray,
        outlier_indices: np.ndarray,
        inlier_indices: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
    ):
        """
        Sample stratified pairs for training.

        Pair distribution:
        - batch_size // 4: (a,a) pairs with target s_a_a
        - batch_size // 4: (a,u) pairs with target s_a_u
        - batch_size // 2: (u,u) pairs with target s_u_u

        Args:
            X_train: Training features
            outlier_indices: Indices of anomalies
            inlier_indices: Indices of normal samples
            batch_size: Batch size
            rng: Random state

        Returns:
            X_left: Left samples (batch_size, n_features)
            X_right: Right samples (batch_size, n_features)
            y_batch: Target scores (batch_size,)
        """
        n_aa = batch_size // 4
        n_au = batch_size // 4
        n_uu = batch_size // 2

        # (a,a) pairs
        idx_a_left = rng.choice(outlier_indices, n_aa, replace=True)
        idx_a_right = rng.choice(outlier_indices, n_aa, replace=True)

        # (a,u) pairs
        idx_au_a = rng.choice(outlier_indices, n_au, replace=True)
        idx_au_u = rng.choice(inlier_indices, n_au, replace=True)

        # (u,u) pairs
        idx_u_left = rng.choice(inlier_indices, n_uu, replace=True)
        idx_u_right = rng.choice(inlier_indices, n_uu, replace=True)

        # Combine indices
        idx_left = np.concatenate([idx_a_left, idx_au_a, idx_u_left])
        idx_right = np.concatenate([idx_a_right, idx_au_u, idx_u_right])

        # Create target scores
        y_batch = np.concatenate([
            np.full(n_aa, self.s_a_a),
            np.full(n_au, self.s_a_u),
            np.full(n_uu, self.s_u_u),
        ])

        # Shuffle
        shuffle_idx = rng.permutation(len(y_batch))
        idx_left = idx_left[shuffle_idx]
        idx_right = idx_right[shuffle_idx]
        y_batch = y_batch[shuffle_idx]

        return X_train[idx_left], X_train[idx_right], y_batch

    def _predict_scores(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Predict anomaly scores for test samples.

        For each test sample, compute score by comparing with reference samples
        from training set (both anomalies and normal samples).

        Args:
            model: Trained PReNet model
            X_test: Test features
            rng: Random state

        Returns:
            Anomaly scores (n_test,)
        """
        outlier_indices = np.where(self.y_train == 1)[0]
        inlier_indices = np.where(self.y_train == 0)[0]

        scores = []
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            for i in range(len(X_test)):
                # Sample reference anomalies and normal samples
                idx_a = rng.choice(outlier_indices, self.num_ref, replace=True)
                idx_u = rng.choice(inlier_indices, self.num_ref, replace=True)

                X_train_a = self.X_train_tensor[idx_a]
                X_train_u = self.X_train_tensor[idx_u]

                # Repeat test sample
                x_test_repeated = X_test_tensor[i].unsqueeze(0).repeat(self.num_ref, 1)

                # Compare with anomalies: score(a, x)
                score_a_x = model(X_train_a, x_test_repeated)

                # Compare with normal: score(x, u)
                score_x_u = model(x_test_repeated, X_train_u)

                # Average score
                score = torch.mean(score_a_x + score_x_u).item()
                scores.append(score)

        return np.array(scores)
