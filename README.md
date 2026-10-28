# USE-AD

**Universal Semantic Enhancement for Weakly Supervised Anomaly Detection**

[![arXiv](https://img.shields.io/badge/arXiv-2610.23765-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.23765)

USE-AD combines frozen TabPFN representations with domain-specific autoencoder features
for weakly supervised anomaly detection on tabular data. ProtoFiLM-Dynamic integrates
the two representations through prototype-conditioned modulation, with RoSAS label
interpolation during training and predictive dynamic fusion at inference.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Simon-zsy/USE-AD.git
cd USE-AD
uv sync --extra baselines --extra plots
```

## Data and Pretrained Weights

Prepare ADBench datasets in `.npz` format with feature array `X` and binary label
array `y` (`0`: normal, `1`: anomaly). Supply the dataset directory and a TabPFN
checkpoint using `--dataset-dir` and `--checkpoint`.

Set `--embedding-dim` to the checkpoint's embedding dimension (default: `512`).
Checkpoint compatibility and dataset requirements are described in the
[configuration notes](docs/DIFFERENCES.md).

## Experiments

Run the main comparison:

```bash
uv run scripts/run_paper.py \
  --suite main \
  --dataset-dir /path/to/Classical \
  --checkpoint /path/to/model.ckpt \
  --device cuda:0 \
  --execute
```

Omit `--execute` to display the experiment configuration without starting training.
Use `--suite` to select an experiment:

| Suite | Experiment |
| --- | --- |
| `main` | USE-AD and seven baselines |
| `ablation` | Component ablations |
| `label-efficiency` | Labeled-anomaly counts of 8, 15, 30, 60, and 120 |
| `sensitivity` | RoSAS loss-weight sensitivity |

The baselines are Random Forest, CatBoost, DevNet, FEAWAD, PReNet, Deep SAD, and TabPFN.

### Default Configuration

| Parameter | Value |
| --- | --- |
| Autoencoder pretraining epochs | 100 |
| Joint training epochs | 30 |
| Adam learning rate | 0.001 |
| Score margin | 5 |
| RoSAS loss weight | 0.5 |
| Labeled anomalies | 8 |
| Independent runs | 10 |
| Train/test split | 80/20, stratified |
| Unlabeled contamination target | 2% |

Experiment definitions are provided in `exp_config.yaml`; global settings are in
`overrides.yaml`. See the [method-to-code mapping](docs/PAPER_CODE_MAP.md) for the
implementation of each component.

## Results

Aggregate per-dataset AUROC and AUPRC, then plot the comparison:

```bash
uv run scripts/summarize_results.py results_paper/results_8 --out summary/main
uv run scripts/plot_raw_results.py summary/main/macro_common_datasets.csv
```

The summary reports valid run counts and computes macro averages over datasets
with complete results for all selected methods.

## Citation

If you use USE-AD in your research, please cite:

```bibtex
@misc{zhou2026universalsemanticenhancement,
  title         = {Universal Semantic Enhancement for Weakly Supervised Anomaly Detection},
  author        = {Siyu Zhou and Yuqin Xie and Hao Yuan and Yingjie Zhou},
  year          = {2026},
  eprint        = {2610.23765},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.23765}
}
```
