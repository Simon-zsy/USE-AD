# Validation results

- All 36 Python files parsed successfully.
- 9 focused CPU tests passed, including an end-to-end synthetic benchmark/CSV check,
  equation-level fusion/residual/PDF checks, dimension mismatch detection, distinct
  label budget checks, raw metric aggregation and train/test contamination isolation.
- All 24 experiment configurations validated and their method instances constructed.
- Benchmark help, main experiment plan and label-efficiency plan passed.
- Real existing checkpoint smoke passed only with the explicit 192-dimensional
  variant: 24 training samples, 8 inference samples, 4 input features, one AE epoch,
  one joint epoch, one batch per epoch, batch size 4 and one TabPFN estimator. RoSAS
  pseudo-sample embeddings were recomputed with frozen TabPFN, and PDF was enabled.
- The paper-default 512-dimensional run was correctly rejected with the existing
  192-dimensional checkpoint. Its failure is retained in the server-local audit record.
- Source file hashes were rechecked against the extraction provenance and match.
- No full paper experiment was executed. No manuscript metric is claimed reproduced.

Public evidence: real_checkpoint_smoke.json and source_manifest.json.
Detailed command output and incompatibility tracebacks remain in the server-local audit records.
