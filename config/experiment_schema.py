"""Experiment configuration schema."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Single experiment configuration."""

    name: str = Field(..., description="Experiment name (used for output files)")
    method: str = Field(..., description="Method name to use")
    args: Dict[str, Any] = Field(default_factory=dict, description="Method arguments")


class ExperimentsConfig(BaseModel):
    """Root experiments configuration schema."""

    experiments: List[ExperimentConfig] = Field(
        default_factory=list, description="List of experiments to run"
    )

    def get_experiment(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment by name."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None

    def list_experiment_names(self) -> List[str]:
        """List all experiment names."""
        return [exp.name for exp in self.experiments]


def load_experiments_config(path: Path) -> ExperimentsConfig:
    """Load experiments configuration from YAML file."""
    import yaml

    if not path.exists():
        return ExperimentsConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return ExperimentsConfig.model_validate(data)
