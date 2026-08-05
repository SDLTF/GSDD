# GSDD-v0.5.0

Graph Spectral Discrepancy Defense — paired backdoor-specific difference-in-differences diagnostic.

## What changed

Earlier rounds showed that a trigger can be spectrally conspicuous even when the model has not learned a functional backdoor. v0.5 therefore trains four controlled GCNs inside each seed and compares models on identical graphs:

$$
D_{\mathrm{DID}}=[T_{\mathrm{full}}-T_{\mathrm{trigger-only}}]-[T_{\mathrm{label-only}}-T_{\mathrm{none}}]
$$

This is designed to remove:

- the trigger's direct input-frequency signature
- ordinary dirty-label learning without a trigger
- model initialization differences
- training-RNG differences within each pair

See `docs/ROUND5_PROTOCOL.md` for definitions and success criteria.

## Install or update

A full package and an overlay patch are provided. When applying the patch, copy its contents into the existing v0.4.x project root and overwrite matching files. Existing `data`, `results`, and `artifacts` directories are preserved.

## Environment check

Activate the same virtual environment used for the earlier rounds, then run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Verify CUDA explicitly:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

The formal v0.5 runner refuses to continue if CUDA is unavailable.

## Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v05.ps1
```

The smoke test uses a synthetic graph and `device: auto`. It validates execution only.

## Formal next experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

This runs three Cora seeds:

```text
1027
2026
3407
```

Each seed trains:

- `none`
- `label_only`
- `trigger_only`
- `full`
- one repeated `full` numerical-control model

## Upload after completion

The script automatically creates:

```text
artifacts\gsdd_v05_paired_did_aggregate.zip
artifacts\gsdd_v05_paired_did_multiseed.zip
```

Upload `gsdd_v05_paired_did_aggregate.zip` first. The larger multiseed package is only needed when a seed is anomalous or detailed node-level inspection is required.

## Main output files

Per seed:

```text
SUMMARY.md
summary.json
model_behavior.csv
paired_node_scores.csv
paired_detection_metrics.json
paired_permutation_tests.json
paired_raw_feature_metrics.csv
repeat_control.json
paired_graph_checks.json
roc_paired_did.png
pr_paired_did.png
did_band_profile_layer1.png
did_band_profile_layer2.png
scatter_logit_vs_spectral_did.png
```

Aggregate:

```text
PAIRED_DID_SUMMARY.md
paired_did_runs.csv
paired_did_summary_stats.csv
paired_did_success_criteria.csv
duplicate_run_candidates.csv
```

## Interpretation guardrail

A high `did_target_logit_abs` confirms that the paired design captures the learned target association. It does not prove a spectral mechanism. The spectral hypothesis is supported only if scale-invariant scores such as `did_shape_l2` or `did_distribution_l2` remain stable and significant across seeds.
