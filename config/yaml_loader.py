"""YAML configuration loader."""

from pathlib import Path

import yaml

from .config_schema import YAMLConfig


class YAMLConfigLoader:
    """Loader for YAML configuration files."""

    @staticmethod
    def load(config_path: str) -> YAMLConfig:
        """Load and validate YAML config file."""
        path = Path(config_path)

        # Return empty config if file doesn't exist
        if not path.exists():
            return YAMLConfig()

        # Load YAML
        with open(path) as f:
            raw_data = yaml.safe_load(f)

        # Validate with Pydantic
        if raw_data is None:
            return YAMLConfig()

        return YAMLConfig.model_validate(raw_data)
