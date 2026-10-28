"""Random Forest method for anomaly detection using AutoGluon."""

from typing import Any, Dict

from pydantic import Field

from .autogluon_base import AutoGluonBaseConfig, AutoGluonBaseMethod


class RandomForestConfig(AutoGluonBaseConfig):
    """Configuration for Random Forest method."""

    n_estimators: int = Field(default=100, description="Number of trees")
    max_depth: int = Field(default=None, description="Maximum tree depth (None for unlimited)")
    criterion: str = Field(default="gini", description="Split criterion (gini or entropy)")


class RandomForestMethod(AutoGluonBaseMethod):
    """Random Forest method for anomaly detection using AutoGluon."""

    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.n_estimators = config.n_estimators
        self.max_depth = config.max_depth
        self.criterion = config.criterion

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def model_key(self) -> str:
        return "RF"

    def get_hyperparameters(self) -> Dict[str, Any]:
        params = {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
        }
        if self.max_depth is not None:
            params["max_depth"] = self.max_depth
        return {self.model_key: params}
