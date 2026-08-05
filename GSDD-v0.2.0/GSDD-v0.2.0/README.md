# GSDD-v0.2.0

Graph Spectral Discrepancy Defense diagnostic experiment, calibration revision.

This version follows the first Cora run of GSDD-v0.1, where the raw spectral coordinates contained strong signal but the label-conditioned median/MAD calibration reversed when the target-label group became 50% poisoned.

## What v0.2 tests

The same raw components are retained:

1. local structural spectral moments
2. input graph-frequency distribution
3. supervised-versus-DGI Jensen–Shannon discrepancy
4. same-label spectral relation contraction
5. supervised-versus-DGI band-transfer discrepancy

Five calibration families are now compared:

- `legacy`: v0.1 label/degree-conditioned MAD
- `global_mad`: label-free two-sided robust deviation
- `global_ecdf`: label-free two-sided empirical-tail score
- `trimmed`: remove globally suspicious samples, then rebuild class references
- `hybrid`: rank fusion of global MAD, global ECDF, and trimmed scores

The experiment also writes `raw_feature_metrics.csv`. This file uses poison ground truth only for scientific diagnosis: it reports whether each primitive coordinate separates clean and poisoned nodes and whether poisoned values are high or low. It is not used to build operational scores.

## Recommended run order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke.ps1
```

### 2. Recommended next experiment

This uses four dirty-label poisoned training nodes. The observed target-label contamination is therefore expected to remain well below the 50% breakdown point seen in v0.1.

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

The result is automatically packaged under `artifacts\` without `.pt` model files.

### 3. Fast contamination sweep

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_calibration_sweep_fast.ps1
```

This runs poison counts 4, 8, 12, and 20 with the shorter Cora configuration, aggregates the runs, and creates one result-set ZIP.

### 4. Clean-label diagnostic

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_cora_clean_label.ps1
```

## Other scripts

```text
scripts/run_cora_fast.ps1
scripts/run_cora_low_poison.ps1
scripts/run_cora_medium_poison.ps1
scripts/run_cora_high_poison.ps1
scripts/run_cora_clean_label.ps1
scripts/run_multiseed.ps1
scripts/package_latest_result.ps1
scripts/package_result_set.ps1
```

Every main run script automatically packages its latest matching result. To package manually:

```powershell
powershell.exe -ExecutionPolicy Bypass `
    -File .\scripts\package_latest_result.ps1 `
    -NamePrefix gsdd_v02_cora_low_poison `
    -Force
```

## Key outputs

```text
SUMMARY.md
summary.json
detection_metrics.json
node_scores.csv
raw_feature_metrics.csv
label_contamination_diagnostic.csv
roc_calibration_comparison.png
pr_calibration_comparison.png
raw_feature_signal_audit.png
band_transfer_layer1.png
band_transfer_layer2.png
```

## How to interpret the next result

First check that the attack was learned:

- `triggered_test_asr` should be high enough to make detection meaningful
- clean accuracy should remain usable

Then compare:

- `legacy_combined`
- `global_mad_combined`
- `global_ecdf_combined`
- `trimmed_combined`
- `hybrid_combined`

Finally inspect `raw_feature_metrics.csv`:

- strong primitive AUROC but weak operational scores means calibration still fails
- strong primitive and operational AUROC means the spectral signal survives unsupervised calibration
- weak primitive AUROC means the relevant spectral hypothesis is unsupported in that setting

## Environment

The implementation does not require PyTorch Geometric. Runtime requirements are listed in `requirements-runtime.txt`.

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Scope

This remains a diagnostic research prototype. It is not yet a finalized defense and should not be evaluated only by one seed, one attack, or one dataset.
