"""CatBoost method for anomaly detection using AutoGluon."""

from typing import Any, Dict

from pydantic import Field

from .autogluon_base import AutoGluonBaseConfig, AutoGluonBaseMethod


class CatBoostConfig(AutoGluonBaseConfig):
    """Configuration for CatBoost method."""

    iterations: int = Field(default=1000, description="Number of boosting iterations")
    learning_rate: float = Field(default=0.03, description="Learning rate")
    depth: int = Field(default=6, description="Tree depth")


class CatBoostMethod(AutoGluonBaseMethod):
    """CatBoost method for anomaly detection using AutoGluon."""

    def __init__(self, config: CatBoostConfig):
        super().__init__(config)
        self.iterations = config.iterations
        self.learning_rate = config.learning_rate
        self.depth = config.depth

    @property
    def name(self) -> str:
        return "catboost"

    @property
    def model_key(self) -> str:
        return "CAT"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            self.model_key: {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "depth": self.depth,
            }
        }
