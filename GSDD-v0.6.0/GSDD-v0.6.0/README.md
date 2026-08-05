# GSDD-v0.6.0

Graph Spectral Discrepancy Defense — CUDA-only attack-family generalization.

This version extends the paired spectral difference-in-differences experiment from one fixed trigger to three trigger-generation mechanisms:

- `fixed_rare_clique`
- `ugba_style_adaptive`
- `dpgba_style_distribution`

The learned families are self-contained, mechanism-faithful adapters inspired by UGBA and DPGBA. They are not verbatim official-repository reproductions; keep the `_style_` names in all reports.

## What is new

- victim-conditioned sparse adaptive trigger generator
- target-class prototype-mixture distribution-preserving generator
- automatic clique/star/chain/cycle topology search
- provisional poisoned surrogate used to optimize learned triggers
- strict four-model paired control for every attack family
- three-seed cross-family aggregation
- attack-generation stealth diagnostics
- CUDA hard requirement in both code and configuration
- automatic result packaging

## CUDA requirement

There is no CPU fallback.

Run:

```powershell
python check_cuda_required.py
```

The command must report:

```text
"cuda_available": true
```

If it reports false, install a CUDA-enabled PyTorch build before running any v0.6 experiment.

## Recommended order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v06.ps1
```

### 2. Full next-stage experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

The full run executes three attack families with seeds 1027, 2026, and 3407.

### 3. Upload the aggregate

```text
artifacts\attack_generalization_aggregate.zip
```

The full node-level archive is:

```text
artifacts\gsdd_v06_attack_generalization_multiseed.zip
```

## Main outputs

The aggregate contains:

```text
ATTACK_GENERALIZATION_SUMMARY.md
attack_generalization_runs.csv
attack_generalization_group_stats.csv
attack_generalization_success_criteria.csv
attack_generalization_family_pass_rates.csv
attack_generalization_run_candidates.csv
duplicate_run_candidates.csv
```

Each individual run contains:

```text
SUMMARY.md
summary.json
attack_diagnostics.json
attack_generator_history.csv       # learned families only
attack_plan.pt
model_behavior.csv
paired_node_scores.csv
paired_detection_metrics.json
paired_permutation_tests.json
paired_raw_feature_metrics.csv
```

## Interpretation

The central question is not simply whether a trigger can be detected. It is whether the scale-invariant backdoor-specific interaction

$$
D_{\mathrm{DID}}
=
[T_{\mathrm{full}}-T_{\mathrm{trigger-only}}]
-
[T_{\mathrm{label-only}}-T_{\mathrm{none}}]
$$

remains predictive when the trigger construction changes.

Read [`docs/ROUND6_PROTOCOL.md`](docs/ROUND6_PROTOCOL.md) before interpreting results.

## Official inspirations

- UGBA official repository: https://github.com/ventr1c/UGBA
- DPGBA official repository: https://github.com/zzwjames/DPGBA

The official UGBA repository uses an older PyTorch/PyG environment. v0.6 intentionally implements the relevant mechanisms inside the existing GSDD stack so that graph loading, splits, GCN training, and paired controls stay identical across families.
