"""Benchmark script for anomaly detection methods."""

import argparse
import gc
import logging
import resource
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from config import (
    MethodRegistry,
    add_common_args,
    discover_datasets,
    load_and_merge_config,
    load_experiments_config,
)
from dataset import load_dataset, prepare_weakly_supervised_data
from methods import BaseMethod


def create_file_logger(
    method_name: str, log_dir: Path, timestamp: str
) -> logging.Logger:
    """
    Create a file-only logger.

    Args:
        method_name: Name of the method
        log_dir: Directory for log files
        timestamp: Timestamp string

    Returns:
        Logger instance
    """
    logger = logging.getLogger(f"{method_name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # File handler
    log_file = log_dir / f"{timestamp}_{method_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def save_results_csv(
    results: List[Dict[str, Any]],
    method_name: str,
    dataset_name: str,
    output_dir: Path,
    timestamp: str,
):
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        method_name: Name of the method
        dataset_name: Name of the dataset
        output_dir: Output directory
        timestamp: Timestamp string
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df["run"] = range(len(results))

    # Calculate average
    avg_row = {
        "run": -1,
        "auc_roc": df["auc_roc"].mean(),
        "auc_pr": df["auc_pr"].mean(),
        "runtime_seconds": df["runtime_seconds"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save to CSV
    csv_file = output_dir / f"{timestamp}_{method_name}_{dataset_name}.csv"
    df.to_csv(csv_file, index=False)


def initialize_experiments(
    args: argparse.Namespace,
    config,
    console: Console,
) -> Dict[str, Tuple[str, BaseMethod]]:
    """
    Initialize experiments based on --experiments flag.

    If experiment name is defined in exp_config.yaml, use that config.
    If experiment name matches a method name and no config exists, use defaults.

    Args:
        args: Parsed command line arguments
        config: Merged YAMLConfig
        console: Rich console for output

    Returns:
        Dict mapping experiment_name -> (method_name, method_instance)
    """
    experiments = {}
    exp_names = [x.strip() for x in args.experiments.split(",")]
    exp_config = load_experiments_config(Path(args.exp_config))

    console.print(f"\n[bold cyan]Initializing {len(exp_names)} experiments...[/bold cyan]")

    for exp_name in exp_names:
        exp = exp_config.get_experiment(exp_name)

        if exp:
            method_name = exp.method
            method_args = exp.args
        elif exp_name in MethodRegistry.list_methods():
            method_name = exp_name
            method_args = {}
        else:
            console.print(f"  [red]✗[/red] {exp_name}: not found in exp_config.yaml and not a valid method name")
            console.print(
                f"\n[yellow]Available methods:[/yellow] {', '.join(MethodRegistry.list_methods())}"
            )
            console.print(
                f"[yellow]Defined experiments:[/yellow] {', '.join(exp_config.list_experiment_names())}"
            )
            raise ValueError(f"Unknown experiment: {exp_name}")

        try:
            config_class = MethodRegistry.get_config_class(method_name)
            merged_args = {**method_args, "device": config.device}
            method_config = config_class.model_validate(merged_args)
            method_instance = MethodRegistry.create_method(method_name, method_config)
            experiments[exp_name] = (method_name, method_instance)
            console.print(f"  [green]✓[/green] {exp_name} (method: {method_name})")
        except ValueError as e:
            console.print(f"  [red]✗[/red] {exp_name}: {e}")
            raise

    return experiments


def has_existing_results(output_dir: Path, exp_name: str, dataset_name: str) -> bool:
    suffix = f"_{exp_name}_{dataset_name}.csv"
    return any(f.name.endswith(suffix) for f in output_dir.glob("*.csv"))


def main():
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection methods")
    parser = add_common_args(parser)
    args = parser.parse_args()

    # Raise file descriptor limit to avoid "Too many open files" with multiprocessing

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(65536, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    console = Console()

    # Load and merge configuration
    config = load_and_merge_config(args)

    console.print(f"\n[bold]Using device: {config.device}[/bold]")

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Discover datasets
    datasets = discover_datasets(
        config.dataset_dir, config.datasets, config.exclude_datasets, config.min_anomalies
    )

    console.print(f"\n[bold]Found {len(datasets)} datasets[/bold]")
    if not datasets:
        raise SystemExit("No eligible datasets found; check the path, selection and minimum anomaly count")
    if config.exclude_datasets:
        console.print(
            f"[yellow]Excluded datasets: {', '.join(config.exclude_datasets)}[/yellow]"
        )

    # Initialize experiments (handles both --experiments and --methods)
    try:
        experiments = initialize_experiments(args, config, console)
    except ValueError:
        raise SystemExit(2)

    # Track all results for summary
    all_results = []
    failures = 0

    # Multi-level progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Level 1: Experiments
        exp_task = progress.add_task(
            "[cyan]Experiments", total=len(experiments) * len(datasets)
        )

        for exp_name, (method_name, method) in experiments.items():
            # Setup logger
            logger = create_file_logger(exp_name, config.log_dir, timestamp)
            method.set_logger(logger)

            logger.info(f"Starting benchmark for experiment: {exp_name} (method: {method_name})")
            logger.info(f"Configuration: {method.config.model_dump()}")

            for dataset_path in datasets:
                dataset_name = dataset_path.stem

                if args.skip_existing and has_existing_results(config.output_dir, exp_name, dataset_name):
                    console.print(f"  [yellow]Skipping {exp_name} - {dataset_name} (results exist)[/yellow]")
                    progress.update(exp_task, advance=1)
                    continue

                # Log dataset start
                logger.info("=" * 60)
                logger.info(f"Starting dataset: {dataset_name}")
                logger.info("=" * 60)

                # Level 2: Runs
                run_task = progress.add_task(
                    f"[green]{exp_name} - {dataset_name}", total=config.runs
                )

                results = []

                try:
                    for run_idx in range(config.runs):
                        logger.info(f"[{dataset_name}] Run {run_idx + 1}/{config.runs}")
                        # Load data
                        X, y = load_dataset(str(dataset_path))

                        # Preprocess
                        data = prepare_weakly_supervised_data(
                            X,
                            y,
                            known_outliers=config.known_outliers,
                            cont_rate=config.cont_rate,
                            test_size=0.2,
                            random_seed=42 + run_idx,
                            max_train_samples=config.max_train_samples,
                            contamination_policy=config.contamination_policy,
                        )

                        # Log data info
                        n_train = len(data["train_x"])
                        n_outliers = np.sum(data["train_y"] == 1)
                        n_inliers = np.sum(data["train_y"] == 0)
                        logger.info(
                            f"[{dataset_name}] Training set size: {n_train} "
                            f"(outliers: {n_outliers}, inliers: {n_inliers})"
                        )

                        logger.info(f"Actual unlabeled contamination: {data['actual_cont_rate']:.8f}")
                        np.random.seed(42 + run_idx)
                        torch.manual_seed(42 + run_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(42 + run_idx)

                        # Run method
                        result = method.run(
                            data["train_x"],
                            data["train_y"],
                            data["test_x"],
                            data["test_y"],
                            run_idx,
                        )

                        results.append(result)

                        # Log run result
                        logger.info(
                            f"[{dataset_name}] Run {run_idx + 1}/{config.runs} completed: "
                            f"AUC-ROC={result['auc_roc']:.4f}, "
                            f"AUC-PR={result['auc_pr']:.4f}, "
                            f"Time={result['runtime_seconds']:.2f}s"
                        )

                        progress.update(run_task, advance=1)

                    # Save results (use exp_name for file naming)
                    save_results_csv(
                        results, exp_name, dataset_name, config.output_dir, timestamp
                    )

                    # Track for summary
                    avg_result = {
                        "experiment": exp_name,
                        "method": method_name,
                        "dataset": dataset_name,
                        "auc_roc": sum(r["auc_roc"] for r in results) / len(results),
                        "auc_pr": sum(r["auc_pr"] for r in results) / len(results),
                        "runtime": sum(r["runtime_seconds"] for r in results)
                        / len(results),
                    }
                    all_results.append(avg_result)

                    logger.info(
                        f"[{dataset_name}] Dataset completed - Average over {config.runs} runs: "
                        f"AUC-ROC={avg_result['auc_roc']:.4f}, "
                        f"AUC-PR={avg_result['auc_pr']:.4f}, "
                        f"Avg Time={avg_result['runtime']:.2f}s"
                    )
                    logger.info("=" * 60)

                except Exception as e:
                    failures += 1
                    logger.error(f"[{dataset_name}] Error: {str(e)}", exc_info=True)
                    console.print(f"[red]Error on {dataset_name}: {str(e)}[/red]")

                progress.remove_task(run_task)
                progress.update(exp_task, advance=1)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    # Print summary table
    console.print("\n[bold]Summary Results[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment", style="cyan")
    table.add_column("Method", style="blue")
    table.add_column("Dataset", style="green")
    table.add_column("AUC-ROC", justify="right")
    table.add_column("AUC-PR", justify="right")
    table.add_column("Runtime (s)", justify="right")

    for result in all_results:
        table.add_row(
            result["experiment"],
            result["method"],
            result["dataset"],
            f"{result['auc_roc']:.4f}",
            f"{result['auc_pr']:.4f}",
            f"{result['runtime']:.2f}",
        )

    console.print(table)
    console.print(f"\n[bold green]Results saved to: {config.output_dir}[/bold green]")
    console.print(f"[bold green]Logs saved to: {config.log_dir}[/bold green]")
    if failures:
        raise SystemExit(f"{failures} experiment/dataset jobs failed; see logs. No incomplete CSVs were saved.")


if __name__ == "__main__":
    main()
