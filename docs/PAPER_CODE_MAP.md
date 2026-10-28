# Paper-to-code map

Source document: user-supplied `Springer_Lecture_Notes_in_Computer_Science (21).pdf`,
"Universal Semantic Enhancement for Weakly Supervised Anomaly Detection on Tabular Data".

| Paper | Extracted implementation |
| --- | --- |
| Eq. 1, frozen TabPFN, 512-dimensional cached embeddings | `methods/tabpfn_feawad.py`, `_embeddings` and Phase 2 in `fit` |
| Eqs. 2–4, AE hidden, residual direction, L2 error | `model/autoencoder.py`, `model/tabpfn_enhanced_deviation_network.py` |
| Eqs. 5–8, fixed prototypes, soft distance attention, FiLM | `model/fusion/dynamic_proto_film_fusion.py`; prototypes set in `fit` |
| Eqs. 9–11, 256/32 scoring MLP, error at every layer | `model/tabpfn_enhanced_deviation_network.py` |
| Eqs. 12–14, deviation and reconstruction margins | `model/losses.py`, `FeawadDevLoss` |
| Eqs. 15–16, one-to-one interpolations and continuous targets | `methods/tabpfn_feawad.py`, joint-training loop |
| Eqs. 17–19, inference PDF | `methods/tabpfn_feawad.py`, `pdf_fuse` |
| Algorithm 1, three phases and fixed 100/30 epochs | `methods/tabpfn_feawad.py`, `fit` |
| Seven external baselines, Sec. 4.1 | Seven baseline modules in `methods/` and their dependency modules |
| Main results, Sec. 4.2 | `scripts/run_paper.py --suite main` |
| Label efficiency, Sec. 4.3 | `scripts/run_paper.py --suite label-efficiency` |
| Table 3, seven component ablations | `scripts/run_paper.py --suite ablation`, `exp_config.yaml` |
| Lambda sensitivity, Sec. 4.5 | `scripts/run_paper.py --suite sensitivity`, `lambda-*` configurations |

The rewritten core method exposes `fit` and `predict_score`, so test labels are
not needed for fitting. Optional penultimate embeddings can be exported for t-SNE.
The original t-SNE script is not included because it retrained a different 50-epoch
configuration. The package supplies fresh raw-result plots, rather than asserting
pixel or numerical equality with the paper's figures.
