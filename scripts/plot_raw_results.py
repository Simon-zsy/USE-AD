"""Plot an unmodified AUROC/AUPRC summary produced by summarize_results.py."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="macro_common_datasets.csv")
    parser.add_argument("--out", type=Path, default=Path("figures/raw_comparison.pdf"))
    args = parser.parse_args()
    data = pd.read_csv(args.summary)
    if data.empty:
        raise SystemExit("No complete common datasets to plot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["auc_roc", "auc_pr"]):
        subset = data[data.metric == metric]
        ax.bar(subset.experiment, subset.macro_mean)
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=60)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
