"""Common CLI utilities for benchmark and related scripts."""

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from .config_schema import YAMLConfig
from .yaml_loader import YAMLConfigLoader


def discover_datasets(
    dataset_path: Path,
    filter_names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    min_anomalies: int = 30,
) -> List[Path]:
    """
    Discover all datasets in the directory.

    Args:
        dataset_path: Path to dataset directory
        filter_names: Optional list of dataset names to filter (include only these)
        exclude_names: Optional list of dataset names to exclude

    Returns:
        List of dataset paths
    """
    datasets = []

    # Find all .npz and .csv files
    datasets.extend(dataset_path.glob("*.npz"))
    datasets.extend(dataset_path.glob("*.csv"))

    # Filter by name if specified
    if filter_names:
        filter_set = set(filter_names)
        datasets = [d for d in datasets if any(f in d.stem for f in filter_set)]

    # Exclude datasets if specified
    if exclude_names:
        exclude_set = set(exclude_names)
        datasets = [d for d in datasets if not any(e in d.stem for e in exclude_set)]

    from dataset import anomaly_count
    return sorted(d for d in datasets if anomaly_count(d) >= min_anomalies)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common arguments used by benchmark and related scripts.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Updated parser
    """
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="overrides.yaml",
        help="YAML config file (default: overrides.yaml)",
    )

    # Dataset arguments (CLI overrides YAML)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to dataset directory (overrides YAML)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of dataset names (overrides YAML)",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        help="Comma-separated list of dataset names to exclude (overrides YAML)",
    )

    # Experiment arguments
    parser.add_argument(
        "--experiments",
        type=str,
        required=True,
        help="Comma-separated experiment names (required)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="exp_config.yaml",
        help="Experiment config file (default: exp_config.yaml)",
    )

    parser.add_argument("--min-anomalies", type=int, help="Minimum anomaly count per dataset (paper: 30)")
    parser.add_argument("--contamination-policy", choices=["strict", "available"],
                        help="strict rejects unavailable 2%% contamination; available records a deviation")

    # Run arguments (CLI overrides YAML)
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of runs per dataset (overrides YAML)",
    )
    parser.add_argument(
        "--known-outliers",
        type=int,
        help="Number of labeled anomalies in training (overrides YAML)",
    )
    parser.add_argument(
        "--cont-rate",
        type=float,
        help="Contamination rate (overrides YAML)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Maximum training samples (overrides YAML)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Results directory (overrides YAML)",
    )
    parser.add_argument("--log-dir", type=str, help="Logs directory (overrides YAML)")

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        help="Device: 'auto', 'cuda', 'cpu', or 'cuda:0' (default: auto)",
    )

    # Skip existing results
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip experiments that already have results in the output directory",
    )

    return parser


def load_and_merge_config(args: argparse.Namespace) -> YAMLConfig:
    """
    Load YAML config and merge with CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Merged YAMLConfig object
    """
    # Load YAML config (provides all defaults)
    yaml_config = YAMLConfigLoader.load(args.config)

    # Build CLI overrides dict (only contains user-specified values)
    cli_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k != "config"
    }

    # Resolve device (special handling)
    device_arg = cli_overrides.pop("device", None)
    if device_arg == "auto" or device_arg is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    # Store resolved device
    cli_overrides["device"] = device

    # Merge configs: YAML defaults < CLI overrides
    merged_config_dict = {**yaml_config.model_dump(), **cli_overrides}

    # Validate and create final config object
    config = YAMLConfig.model_validate(merged_config_dict)

    return config


def get_dataset_list(config: YAMLConfig) -> List[str]:
    """
    Get list of dataset names to process based on config.

    Args:
        config: YAMLConfig object

    Returns:
        List of dataset names (without extensions)
    """
    # Discover dataset files
    dataset_paths = discover_datasets(
        config.dataset_dir,
        config.datasets,
        config.exclude_datasets,
        config.min_anomalies,
    )

    # Extract dataset names
    dataset_names = [d.stem for d in dataset_paths]

    return dataset_names
