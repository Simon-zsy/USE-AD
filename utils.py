"""
Utility functions for data loading, preprocessing, and evaluation.
"""

import csv
from typing import Optional, Tuple

import numpy as np


def load_csv_dataset(
    csv_path: str,
    feature_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV file (FEAWAD format).

    The CSV format is:
    - First N columns: features
    - Last column: label (0=normal, 1=anomaly)

    Args:
        csv_path: Path to CSV file
        feature_dim: Number of feature columns (optional, auto-detected if None)

    Returns:
        Tuple of (features, labels)
    """
    features = []
    labels = []

    with open(csv_path, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if feature_dim is None:
                # Auto-detect: assume last column is label, rest are features
                features.append(row[:-1])
                labels.append(row[-1])
            else:
                # Use specified feature dimension
                features.append(row[:feature_dim])
                labels.append(row[feature_dim])

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return features, labels


def load_npz_dataset(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from .npz file (ADBench format).

    Args:
        npz_path: Path to .npz file

    Returns:
        Tuple of (features, labels)
    """
    with np.load(npz_path) as data:
        features = data["X"].astype(np.float32)
        labels = data["y"].astype(np.float32)

    return features, labels
