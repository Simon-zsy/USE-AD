"""TabPFN: Direct classification using pre-trained TabPFN model."""

import time
from typing import Dict, Any
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from pydantic import BaseModel, Field
from tabpfn import TabPFNClassifier

from .base import BaseMethod


class TabPFNConfig(BaseModel):
    """Configuration for TabPFN method."""

    model_path: str = Field(
        default="checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt",
        description="Path to TabPFN model checkpoint",
    )
    n_estimators: int = Field(
        default=4, description="Number of TabPFN estimators to ensemble"
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (cuda or cpu)",
    )


class TabPFNMethod(BaseMethod):
    """
    TabPFN: Direct classification using pre-trained TabPFN model.

    Uses TabPFN's few-shot learning capabilities for anomaly detection.
    No training required - just fit and predict with the pre-trained model.
    """

    def __init__(self, config: TabPFNConfig):
        """
        Initialize TabPFN method.

        Args:
            config: TabPFNConfig instance with hyperparameters
        """
        self.config = config
        self.model_path = config.model_path
        self.n_estimators = config.n_estimators
        self.device = config.device

    @property
    def name(self) -> str:
        return "tabpfn"

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run TabPFN method.

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

        self.log(f"Training TabPFN with {self.n_estimators} estimators")
        self.log(
            f"Training samples: {len(X_train)} "
            f"({np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} anomalies)"
        )
        self.log(f"Test samples: {len(X_test)}")

        # Initialize TabPFN classifier
        clf = TabPFNClassifier(
            model_path=self.model_path,
            n_estimators=self.n_estimators,
            device=self.device,
        )

        # Fit on training data (no gradient descent, just caching)
        self.log("Fitting TabPFN on training data...")
        clf.fit(X_train, y_train)

        # Get prediction probabilities
        # Shape: (n_test, 2) - [prob_normal, prob_anomaly]
        self.log("Predicting on test data...")
        probs = clf.predict_proba(X_test)

        # Use probability of anomaly class as anomaly score
        scores = probs[:, 1]

        # Evaluate
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
