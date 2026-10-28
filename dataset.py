"""Train/test separation and weak labels for the paper-oriented implementation.

The paper does not specify what to do when the training fold contains too few
hidden anomalies for 2% contamination. We fail explicitly by default. The
optional 'available' policy reports the achieved rate and is a protocol deviation.
"""
from pathlib import Path
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_npz_dataset, load_csv_dataset

def load_dataset(dataset_path):
    path = Path(dataset_path)
    if path.suffix == ".npz":
        return load_npz_dataset(str(path))
    if path.suffix == ".csv":
        return load_csv_dataset(str(path))
    raise ValueError(f"Unsupported dataset format: {path.suffix}")

def prepare_weakly_supervised_data(x, labels, known_outliers=8, cont_rate=0.02,
        test_size=0.2, random_seed=42, max_train_samples=50000,
        contamination_policy="strict"):
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels containing both 0 and 1")
    if not 0 <= cont_rate < 1 or known_outliers < 1:
        raise ValueError("Require 0 <= cont_rate < 1 and known_outliers >= 1")
    if contamination_policy not in {"strict", "available"}:
        raise ValueError("Unknown contamination policy")
    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=test_size,
        random_state=random_seed, stratify=labels)
    rng = np.random.RandomState(random_seed)
    anomaly = train_idx[labels[train_idx] == 1]
    normal = train_idx[labels[train_idx] == 0]
    if len(anomaly) < known_outliers:
        raise ValueError(f"Requested {known_outliers} distinct labeled anomalies; training fold has {len(anomaly)}")
    selected = rng.choice(anomaly, known_outliers, replace=False)
    hidden_pool = np.setdiff1d(anomaly, selected)
    if max_train_samples <= known_outliers:
        raise ValueError("max_train_samples must exceed known_outliers")
    budget = max_train_samples - known_outliers
    n_normal = min(len(normal), int(np.floor(budget * (1 - cont_rate))))
    n_hidden = int(np.floor(n_normal * cont_rate / (1 - cont_rate)))
    if n_hidden > len(hidden_pool):
        message = (f"2%-style contamination request needs {n_hidden} distinct hidden training anomalies, "
                   f"but only {len(hidden_pool)} remain. No test samples or synthetic anomalies are used.")
        if contamination_policy == "strict":
            raise ValueError(message)
        warnings.warn(message + " Using the available pool; the achieved rate is reported.")
        n_hidden = len(hidden_pool)
    normals = rng.choice(normal, n_normal, replace=False)
    hidden = rng.choice(hidden_pool, n_hidden, replace=False)
    indices = np.concatenate([normals, hidden, selected])
    weak_y = np.concatenate([np.zeros(n_normal + n_hidden), np.ones(known_outliers)])
    order = rng.permutation(len(indices))
    return {"train_x": x[indices[order]], "train_y": weak_y[order],
            "test_x": x[test_idx], "test_y": labels[test_idx],
            "train_indices": indices[order], "test_indices": test_idx,
            "hidden_anomaly_indices": hidden, "labeled_anomaly_indices": selected,
            "n_noise": n_hidden, "actual_cont_rate": n_hidden / max(1, n_normal + n_hidden)}


def anomaly_count(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return int((data["y"] == 1).sum())
    _, y = load_dataset(path)
    return int((y == 1).sum())
