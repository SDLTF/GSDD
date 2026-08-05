# Validation

## Static validation

- All Python files compile successfully with `python -m py_compile`
- New YAML configuration fields load through the dataclass configuration system
- The v0.5 runner remains backward compatible when the reproducibility section is absent

## Local smoke validation

Command:

```text
python run_repro_audit.py --config configs/smoke_v051.yaml --strict
```

Environment: CPU container

Observed behavior:

- two complete paired replicas finished
- `full` ASR was 1.0 in both replicas
- `trigger_only` ASR was 0.0 in both replicas
- initialization hashes matched
- maximum parameter difference was 0
- all node-level score correlations were 1.0
- all operational top-k overlaps were 1.0
- all AUROC and AUPRC differences were 0
- all four primary reproducibility criteria passed

The aggregate script was also tested on the smoke audit and produced `REPRO_AUDIT_AGGREGATE.md`, `repro_audit_runs.csv`, and summary statistics.

## Limitation

No CUDA device is available in the build container. The Windows CUDA launchers and their deterministic behavior must be validated on the user's RTX 5060 Laptop environment. The practical CUDA script deliberately records empirical score stability even when PyTorch warns that a sparse CUDA operation lacks a strict deterministic implementation.
