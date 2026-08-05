# GSDD-v0.6.1

Graph Spectral Discrepancy Defense — binding-aware attack-validity repair.

v0.6 showed that the first learned trigger generators behaved as direct targeted evasion perturbations: clean and label-only control models also had high ASR. v0.6.1 changes the generator objective so that a trigger is rewarded only when it activates a poisoned surrogate while preserving the original prediction of both clean and label-only surrogates on the same triggered graph.

## Formal attack families

- `fixed_rare_clique`
- `ugba_style_binding_aware`
- `dpgba_style_binding_aware`

The learned families are self-contained mechanism adapters. They are not official UGBA or DPGBA repository reproductions, so keep the full `_style_binding_aware` names in reports.

## CUDA is mandatory

Formal scripts and Python entry points both force CUDA. There is no CPU fallback.

```powershell
python check_cuda_required.py
```

The command must report `"cuda_available": true`.

## Generator objective

For the same triggered graph, the generator sees three frozen surrogates:

- clean surrogate
- label-only surrogate
- poisoned surrogate

It minimizes poisoned target loss while penalizing target activation on both control surrogates. The central margin is

$$
M_{\mathrm{bind}}
=
(z_t^{\mathrm{poison}}-z_y^{\mathrm{poison}})
-
\max\left\{
 z_t^{\mathrm{clean}}-z_y^{\mathrm{clean}},
 z_t^{\mathrm{label}}-z_y^{\mathrm{label}}
\right\}
$$

The generator is also constrained by neighborhood homophily, feature-distribution distance, sparsity, and topology density.

## Hard attack-admission gate

A run is considered a valid learned backdoor only when

$$
\operatorname{ASR}_{\mathrm{full}}\ge 0.80
$$

$$
\max\{\operatorname{ASR}_{\mathrm{none}},\operatorname{ASR}_{\mathrm{label}},\operatorname{ASR}_{\mathrm{trigger}}\}\le 0.10
$$

$$
\operatorname{ASR}_{\mathrm{full}}-\operatorname{ASR}_{\mathrm{trigger}}\ge 0.60
$$

Invalid attacks are retained for diagnosis but excluded from spectral-generalization claims.

## Run order

### 1. CUDA smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v061.ps1
```

### 2. Full next-stage experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

The fixed baseline runs all three seeds. Each learned family first runs seed 1027 as a pilot. Seeds 2026 and 3407 run only if the pilot passes the functional-backdoor gate.

### 3. Upload

Upload this file first:

```text
artifacts\attack_validity_repair_aggregate.zip
```

The full node-level archive is:

```text
artifacts\gsdd_v061_attack_validity_repair_multiseed.zip
```

## Main aggregate outputs

```text
ATTACK_VALIDITY_REPAIR_SUMMARY.md
attack_validity_runs.csv
attack_validity_group_stats_all.csv
attack_validity_group_stats_valid_only.csv
attack_validity_success_criteria.csv
attack_validity_family_pass_rates.csv
attack_validity_run_candidates.csv
duplicate_run_candidates.csv
```

Each run also contains `attack_validity.json`, `attack_diagnostics.json`, `attack_generator_history.csv`, model behavior, paired DID metrics, permutation tests, and node-level scores.

Read [`docs/ROUND61_PROTOCOL.md`](docs/ROUND61_PROTOCOL.md) before interpreting results.
