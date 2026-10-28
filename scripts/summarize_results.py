"""Aggregate raw benchmark CSVs without scaling, replacing or inventing metrics."""
import argparse
import json
from pathlib import Path
import re
import pandas as pd

def summarize(results_dir, methods=None):
    latest = {}
    for path in sorted(Path(results_dir).glob("*.csv")):
        match = re.match(r"^(\d{8}_\d{6})_(.+)_(\d+_.+)\.csv$", path.name)
        if not match:
            continue
        timestamp, method, dataset = match.groups()
        if methods is None or method in methods:
            latest[method, dataset] = path
    records = []
    for (method, dataset), path in latest.items():
        frame = pd.read_csv(path)
        if "run" in frame:
            frame = frame[frame["run"] != -1]
        for metric in ["auc_roc", "auc_pr"]:
            values = pd.to_numeric(frame[metric], errors="raise").dropna()
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid {metric} in {path}")
            records.append(dict(experiment=method, dataset=dataset, metric=metric,
                mean=values.mean(), std=values.std(ddof=1), n=len(values), source=path.name))
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--experiments", help="Comma-separated experiment names")
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("summary"))
    args = parser.parse_args()
    methods = args.experiments.split(",") if args.experiments else None
    data = summarize(args.results_dir, methods)
    if data.empty:
        raise SystemExit("No matching result CSVs")
    selected = methods or sorted(data.experiment.unique())
    data["complete"] = data["n"] == args.expected_runs
    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "per_dataset.csv", index=False)
    rows, coverage = [], {}
    for metric in ["auc_roc", "auc_pr"]:
        metric_data = data[(data.metric == metric) & data.complete]
        pivot = metric_data.pivot(index="dataset", columns="experiment", values="mean")
        pivot = pivot.reindex(columns=selected).dropna()
        coverage[metric] = list(pivot.index)
        if pivot.empty:
            continue
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        for method in selected:
            rows.append(dict(experiment=method, metric=metric,
                macro_mean=pivot[method].mean(), average_rank=ranks[method].mean(),
                datasets=len(pivot)))
    pd.DataFrame(rows).to_csv(args.out / "macro_common_datasets.csv", index=False)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(data.to_string(index=False))
    print("Macro comparisons use only common datasets with the requested number of valid runs.")

if __name__ == "__main__":
    main()
