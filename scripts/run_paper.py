"""Plan paper experiments by default; use --execute to train after preflight."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import anomaly_count, load_dataset, prepare_weakly_supervised_data

BASELINES = ["random_forest", "catboost", "devnet", "feawad", "prenet", "deepsad", "tabpfn"]
FIVE_DATASETS = ["1_ALOI", "10_cover", "16_http", "37_Stamps", "42_WBC"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["main", "ablation", "label-efficiency", "sensitivity"], default="main")
    parser.add_argument("--server-config", type=Path, help="Optional local-only YAML with dataset_dir/checkpoint")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--datasets", help="Optional comma-separated exact dataset stems")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Paper: 512. Current server weight returns 192; selecting 192 is a protocol deviation.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_paper")
    parser.add_argument("--contamination-policy", choices=["strict", "available"], default="strict")
    parser.add_argument("--execute", action="store_true", help="Actually train; otherwise print the plan only")
    args = parser.parse_args()
    server = yaml.safe_load(args.server_config.read_text()) if args.server_config else {}
    data_dir = args.dataset_dir or Path(server.get("dataset_dir", ROOT / "data"))
    checkpoint = args.checkpoint or Path(server.get("checkpoint", ROOT / "checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt"))
    files = {p.stem: p for p in sorted(data_dir.glob("*.npz"))}
    files.update({p.stem: p for p in sorted(data_dir.glob("*.csv"))})
    if args.datasets:
        names = args.datasets.split(",")
    elif args.suite == "label-efficiency":
        names = FIVE_DATASETS
    else:
        names = [name for name, path in files.items() if anomaly_count(path) >= 30]
    if not names:
        parser.error("No eligible datasets. Set --dataset-dir or --server-config.")
    if args.embedding_dim < 1:
        parser.error("--embedding-dim must be positive")
    if args.runs < 1:
        parser.error("--runs must be positive")
    experiments = yaml.safe_load((ROOT / "exp_config.yaml").read_text())
    for experiment in experiments["experiments"]:
        if experiment["method"] in {"tabpfn", "tabpfn-feawad"}:
            experiment.setdefault("args", {})["model_path"] = str(checkpoint.resolve())
        if experiment["method"] == "tabpfn-feawad":
            experiment.setdefault("args", {})["expected_tabpfn_dim"] = args.embedding_dim
    if args.suite == "main":
        methods, counts = BASELINES + ["ours"], [8]
    elif args.suite == "ablation":
        methods = ["concat", "static-proto", "dynamic-proto", "rosas-only", "pdf-only", "dynamic-pdf", "ours"]
        counts = [8]
    elif args.suite == "sensitivity":
        methods = [e["name"] for e in experiments["experiments"] if e["name"].startswith("lambda-")]
        counts = [8]
    else:
        methods, counts = BASELINES + ["ours"], [8, 15, 30, 60, 120]
    report = {"suite": args.suite, "datasets": names, "dataset_count": len(names),
        "paper_expected_main_dataset_count": 42, "runs": args.runs, "methods": methods,
        "checkpoint_exists": checkpoint.is_file(), "expected_tabpfn_dim": args.embedding_dim, "contamination_policy": args.contamination_policy,
        "label_counts": counts, "executed": bool(args.execute)}
    print(json.dumps(report, indent=2))
    if args.suite != "label-efficiency" and len(names) != 42:
        print(f"NOTE: using {len(names)} actual datasets; this is not a 42-dataset reproduction.")
    if not args.execute:
        print("Plan only: no training, no model loading, no result files written. Add --execute to run preflight and train.")
        return
    if not checkpoint.is_file():
        parser.error("Checkpoint is missing; obtain the authorized TabPFN checkpoint first")
    # Validate every split/label budget BEFORE starting an expensive experiment.
    errors = []
    for name in names:
        if name not in files:
            errors.append(f"{name}: dataset file missing")
            continue
        x, y = load_dataset(files[name])
        for known in counts:
            for run in range(args.runs):
                try:
                    prepare_weakly_supervised_data(x, y, known_outliers=known,
                        random_seed=42+run, contamination_policy=args.contamination_policy)
                except ValueError as error:
                    errors.append(f"{name}, labels={known}, seed={42+run}: {error}")
    if errors:
        raise SystemExit("Preflight failed; no training started:\n" + "\n".join(errors))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_plan.json").write_text(json.dumps(report, indent=2))
    with tempfile.TemporaryDirectory(prefix="paper-config-") as temporary:
        exp_path = Path(temporary) / "experiments.yaml"
        exp_path.write_text(yaml.safe_dump(experiments, sort_keys=False))
        for known in counts:
            output = args.output_root / f"results_{known}"
            command = [sys.executable, str(ROOT / "benchmark.py"), "--experiments", ",".join(methods),
                "--exp-config", str(exp_path), "--config", str(ROOT / "overrides.yaml"),
                "--dataset-dir", str(data_dir.resolve()), "--datasets", ",".join(names),
                "--min-anomalies", "0" if args.suite == "label-efficiency" else "30",
                "--known-outliers", str(known), "--runs", str(args.runs), "--device", args.device,
                "--contamination-policy", args.contamination_policy,
                "--output-dir", str(output.resolve()), "--log-dir", str((output / "logs").resolve())]
            subprocess.run(command, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
