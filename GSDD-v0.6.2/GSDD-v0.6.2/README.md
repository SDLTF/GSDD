# GSDD-v0.6.2

Graph Spectral Discrepancy Defense — Attack Validity Repair Round 2.

Round 6.1 showed two distinct failure modes. The UGBA-style adapter produced a strong functional interaction but its label-only ASR was slightly above the hard control threshold. The DPGBA-style adapter kept controls low but did not form a strong trigger-target mapping. v0.6.2 repairs these problems separately without relaxing the admission criteria.

## What changed

### UGBA-style

- reduces dirty-label poison count from the previous four-node setting
- screens target classes 0, 1, and 2
- strengthens binding-gap and clean-control penalties
- selects a candidate only through the unchanged functional-backdoor gate

### DPGBA-style

- uses clean-label target-class training victims
- screens poison counts 8, 12, and 16
- uses a mixed target/global prototype bank
- increases alternating generator/surrogate rounds
- preserves clean and label-only predictions during generator optimization

These remain self-contained mechanism adapters rather than verbatim official UGBA or DPGBA repository reproductions.

## CUDA requirement

Formal scripts require a CUDA-enabled PyTorch installation. There is no silent CPU fallback.

```powershell
python check_cuda_required.py
```

## Run order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v062.ps1
```

### 2. Recommended full experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

This command:

1. runs seven seed-1027 repair candidates
2. aggregates and ranks the candidates
3. expands only valid pilot candidates to seeds 2026 and 3407
4. re-aggregates all completed runs
5. packages the aggregate and node-level run set

### 3. Pilot only

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_attack_validity_repair_round2_pilot.ps1
```

Use this when you want to inspect seed 1027 before spending compute on multiseed expansion.

## Upload after completion

Upload this first:

```text
artifacts\attack_validity_repair_round2_aggregate.zip
```

The larger per-run archive is:

```text
artifacts\gsdd_v062_attack_validity_repair_round2_runs.zip
```

## Hard admission gate

A learned attack is valid only when:

- full ASR is at least 0.80
- every control ASR is at most 0.10
- full ASR minus trigger-only ASR is at least 0.60

Near-valid candidates are ranked for diagnosis, but are not automatically expanded and are not used to support spectral-generalization claims.

Read [`docs/ROUND62_PROTOCOL.md`](docs/ROUND62_PROTOCOL.md) before interpreting the results.
