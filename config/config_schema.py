"""Configuration schema with default values."""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class YAMLConfig(BaseModel):
    """Root YAML configuration schema with default values."""
    runs: int = Field(default=10, description='Number of runs per dataset')
    datasets: Optional[list[str]] = Field(default=None, description='List of dataset names (None = all)')
    exclude_datasets: Optional[list[str]] = Field(default=[], description='List of dataset names to exclude')
    dataset_dir: Path = Field(default=Path('data'), description='Path to dataset directory')
    known_outliers: int = Field(default=8, description='Number of labeled anomalies in training')
    cont_rate: float = Field(default=0.02, description='Contamination rate')
    max_train_samples: int = Field(default=50000, description='Maximum training samples')
    min_anomalies: int = Field(default=30, ge=0)
    contamination_policy: str = Field(default='strict', pattern='^(strict|available)$')
    output_dir: Path = Field(default=Path('./results'), description='Results directory')
    log_dir: Path = Field(default=Path('./logs'), description='Logs directory')
    device: str = Field(default='auto', description="Device: 'auto', 'cuda', 'cpu', or 'cuda:X'")

    @field_validator('datasets', mode='before')
    @classmethod
    def parse_datasets(cls, v):
        """Parse comma-separated string to list."""
        if v is None:
            return None
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        return v

    @field_validator('exclude_datasets', mode='before')
    @classmethod
    def parse_exclude_datasets(cls, v):
        """Parse comma-separated string to list."""
        if v is None:
            return None
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        return v

    @field_validator('exclude_datasets')
    @classmethod
    def validate_no_dataset_conflict(cls, v, info):
        """Ensure no dataset appears in both datasets and exclude_datasets."""
        datasets = info.data.get('datasets')
        if datasets is not None and v is not None:
            conflicts = set(datasets) & set(v)
            if conflicts:
                raise ValueError(f"Dataset(s) cannot appear in both 'datasets' and 'exclude_datasets': {', '.join(sorted(conflicts))}")
        return v
