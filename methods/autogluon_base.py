"""Base class for AutoGluon-based anomaly detection methods."""

import shutil
import tempfile
import time
from abc import abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import BaseMethod


class AutoGluonBaseConfig(BaseModel):
    """Base configuration for AutoGluon methods."""

    time_limit: int = Field(default=300, description="Time limit for training in seconds")
    verbosity: int = Field(default=0, description="AutoGluon verbosity level (0-4)")


class AutoGluonBaseMethod(BaseMethod):
    """Base class for AutoGluon-based anomaly detection methods."""

    def __init__(self, config: AutoGluonBaseConfig):
        self.config = config
        self.time_limit = config.time_limit
        self.verbosity = config.verbosity

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_key(self) -> str:
        """Return the AutoGluon model key (e.g., 'XGB', 'CAT', 'GBM', 'RF')."""
        pass

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters dict for AutoGluon. Override in subclasses."""
        return {self.model_key: {}}

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        from autogluon.tabular import TabularPredictor

        start_time = time.time()

        self.log(
            f"Training {self.name} on {len(X_train)} samples "
            f"({np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} anomalies)"
        )
        self.log(f"Test set: {len(X_test)} samples")

        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train.astype(int)

        test_df = pd.DataFrame(X_test)

        temp_dir = tempfile.mkdtemp()
        try:
            predictor = TabularPredictor(
                label="label",
                path=temp_dir,
                problem_type="binary",
                eval_metric="roc_auc",
                verbosity=self.verbosity,
            )

            predictor.fit(
                train_data=train_df,
                hyperparameters=self.get_hyperparameters(),
                time_limit=self.time_limit,
            )

            pred_proba = predictor.predict_proba(test_df)
            if isinstance(pred_proba, pd.DataFrame):
                scores = pred_proba[1].values
            else:
                scores = pred_proba[:, 1]

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
        except Exception as e:
            runtime_seconds = time.time() - start_time
            self.log(f"Failed: {e}")
            return {
                "auc_roc": float("nan"),
                "auc_pr": float("nan"),
                "runtime_seconds": runtime_seconds,
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
