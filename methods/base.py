"""Abstract base class for anomaly detection methods."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel


class BaseMethod(ABC):
    """Abstract base class for anomaly detection methods."""

    logger: Optional[logging.Logger] = None

    def __init__(self, config: BaseModel) -> None:
        super().__init__()

    def set_logger(self, logger: logging.Logger):
        """
        Set logger (called by benchmark).

        Args:
            logger: Logger instance for this method
        """
        self.logger = logger

    def log(self, message: str, level: str = "info"):
        """
        Log message if logger available.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        if self.logger:
            getattr(self.logger, level)(message)

    @abstractmethod
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run anomaly detection method.

        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,) - 0=normal, 1=anomaly
            X_test: Test features (n_test, n_features)
            y_test: Test labels (n_test,) - 0=normal, 1=anomaly
            run_idx: Run index (for random seed variation)

        Returns:
            Dictionary with:
                - auc_roc: AUC-ROC score
                - auc_pr: AUC-PR score
                - runtime_seconds: Execution time in seconds
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return method name."""
        pass
