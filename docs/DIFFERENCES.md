# Paper-first decisions and reproduction limits

The user explicitly requested that the paper take precedence over existing settings.
These changes affect numerical results. This package must not be described as a
bit-for-bit reproduction of old CSVs or proof of the paper's numerical claims.

| Topic | Existing research checkout | Paper-oriented extraction |
| --- | --- | --- |
| Joint epochs | Named fusion experiments set 50 | 30, no test-driven early stopping |
| AE encoder | Extra d→d input layer before 128→64 | d→128→64 as described in Sec. 3.3 |
| Residual | x−reconstruction plus LayerNorm | reconstruction−x, divided by L2 norm (epsilon only for stability) |
| Fusion features | LayerNorm enabled | Disabled to follow the written equations |
| Normal prototype | Mean over all weakly unlabeled examples | Random U0 subset, maximum 1024 samples; paper does not give its size |
| RoSAS | Multiple historical branches, including random all-pairs; cached embeddings mixed linearly | One pseudo-sample per anomaly/unlabeled pair; lambda 0.5; frozen TabPFN computes pseudo-sample embeddings by default |
| PDF | Confidence term 2*abs(s−0.5) plus agreement | abs(s−0.5) plus agreement, as in Eqs. 17–19 |
| Evaluation | Best test AUROC selected epoch and early stopping | Test scores evaluated after fixed training; `fit` accepts no test data |
| Contamination | Synthetic feature swapping from the full anomaly pool, including test-fold anomalies | Hidden anomalies drawn only from training-fold remainder, without replacement |
| Missing labels | Existing loader could keep fewer than requested | Explicit error if distinct label budget is unavailable |
| Main-table generation | `scripts/generate_main_table.py:61–73,100–107` scales raw per-dataset means toward fixed values | Excluded; summaries compute unmodified CSV means and coverage |
| Sensitivity figure | `scripts/plots/plot_sensitivity_lambda.py:19–22` contains fixed metrics | Excluded; no hardcoded performance metrics in runnable reporting |

## Important constraints

- The available inventory has **47** NPZ datasets; **41** satisfy anomaly count >=30.
  The PDF says 57 source datasets and 42 evaluated datasets. Those statements cannot
  be recovered from the current files by applying only the stated rule. The launcher
  reports its actual count; it never labels a 41-dataset run as a 42-dataset run.
- The original 42-result-dataset set includes datasets with fewer than 30 anomalies
  (for example Lymphography has 6 and WBC has 10). It is not the stated threshold rule.
- WBC has 10 total anomalies, so the requested 15/30/60/120 distinct-label settings
  cannot be constructed. Other datasets also lack enough training anomalies for
  the largest label budget or the exact 2% hidden contamination target.
- Strict mode rejects unavailable 2% contamination rather than using test examples,
  synthesizing anomalies, or silently changing the target. `available` is an explicit
  deviation that uses only available training anomalies and reports the actual rate.
- Interpolated TabPFN embeddings are not explicitly explained by the PDF. Recomputing
  frozen features is the default; `rosas_embedding_mode=linear` is a documented faster
  approximation, not claimed to be equivalent. Recomputing is much more expensive
  than the old cached-embedding interpolation and is inconsistent with a literal
  claim that all TabPFN work happens only once; the paper needs to clarify this detail.
- Decoder ReLU activations and sampling defaults are retained from the source because
  the PDF does not completely specify them. The AE pretraining objective is the
  per-sample squared L2 sum stated in Algorithm 1, rather than an average per feature.
- Existing baseline implementations were retained. Shared AutoEncoder architecture
  follows the paper, which also changes the FEAWAD baseline relative to the source.
  Deep SAD and PReNet retain their original method-specific architectures and losses;
  their defaults are not retroactively asserted to match every general statement in the PDF.
- Removing test leakage, changing formulas or training length requires new experiments.
  Existing manuscript metrics have not been validated and were not changed here.

## Historical configuration evidence

`experiments_recorded_reference.yaml` preserves relevant old experiment blocks. It is
documentation, not a runnable configuration for the stricter paper-first core class.
The historical dynamic-PDF experiment and its inspected logs enable PDF but do not
enable RoSAS, whereas the manuscript calls the full method Dynamic+RoSAS+PDF.
The manuscript's 0.9308 value must not be attributed to that historical config merely
because a plotting script labels it as the full model.

No author/affiliation instructions, advisor notes, personal screenshots, environments,
unrelated fusions, cleanup scripts, generated figures, weights or existing result CSVs
were included in the extracted package.


## Measured checkpoint dimension mismatch

A real CPU smoke run on the server's existing
`tabpfn-v2.5-classifier-v2.5_default.ckpt` with TabPFN 6.2.0 measured **192**
embedding dimensions. The supplied paper states **512**. The default configuration
requires 512 and fails explicitly with this checkpoint. It does not pad, project,
or reshape 192 features to pretend to meet the paper specification.

`expected_tabpfn_dim=192` (launcher: `--embedding-dim 192`) is an explicit compatible
variant for the current checkpoint, with the corresponding fusion input/output
dimensions derived from 192. It is not the paper's 512-dimensional setting. A full
paper-default run needs a verified compatible 512-dimensional checkpoint or a
clarification/correction of the manuscript. No alternative checkpoint was downloaded.
