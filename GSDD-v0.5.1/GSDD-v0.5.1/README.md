# GSDD-v0.5.1

Deterministic and empirical reproducibility audit for the backdoor-specific paired spectral difference-in-differences experiment.

## What changed

v0.5 showed strong spectral DID on Cora, but one CUDA repeat produced a large logit difference. v0.5.1 repeats the **entire four-model paired experiment twice** for each seed and compares:

- model parameters and behavior
- node-level DID scores
- Pearson and Spearman correlations
- top-$k$ candidate overlap
- AUROC and AUPRC drift

The package does not assume that enabling a deterministic flag is sufficient. The observed two-replica stability report is the final diagnostic.

## Install over v0.5.0

Extract the incremental patch into the v0.5.0 project root and overwrite matching files, or use the complete package.

## 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v051.ps1
```

The CPU smoke test runs two identical synthetic paired experiments and should report exact equality.

## 2. Main three-seed GPU audit

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

This runs seeds:

```text
1027, 2026, 3407
```

Each seed trains two complete replicas, with four controlled GCNs per replica. The script then aggregates and packages the results.

Upload this file first:

```text
artifacts\repro_audit_aggregate.zip
```

The full result set is also written to:

```text
artifacts\gsdd_v051_repro_multiseed.zip
```

Model checkpoints are excluded from the ZIP by default.

## 3. Strict CUDA diagnosis for seed 3407

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_strict_cuda_seed3407.ps1
```

This uses `warn_only=false`. An unsupported nondeterministic CUDA sparse operation will terminate with an explicit PyTorch error. This is diagnostic and is not required before the main practical audit.

## 4. Strict CPU reference for seed 3407

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_cpu_reference_seed3407.ps1
```

Use this only if the practical GPU audit remains unstable or strict CUDA reports an unsupported sparse operation.

## Main outputs per seed

The audit directory contains:

```text
REPRODUCIBILITY_AUDIT.md
summary.json
score_reproducibility.csv
node_score_reproducibility.csv
model_parameter_reproducibility.csv
model_behavior_reproducibility.csv
repro_scatter_did_shape_l2.png
repro_scatter_did_distribution_l2.png
repro_scatter_did_spectral_hybrid.png
repro_scatter_did_shape_mahalanobis.png
```

## Interpretation

The key question is not whether two runs have exactly identical weights. The decisive test is whether identical runs preserve:

- scale-invariant node-score ranking
- the operational top-$5\%$ candidate set
- AUROC/AUPRC
- the functional backdoor controls

See `docs/ROUND51_PROTOCOL.md` for the full criteria.
