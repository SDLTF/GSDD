# Validation record

## Static validation

The following files passed Python bytecode compilation:

```text
run_gsdd_v05.py
aggregate_paired_did.py
gsdd/paired_diagnostics.py
gsdd/config.py
```

## Synthetic smoke test

Environment:

```text
torch 2.10.0+cpu
synthetic graph: 180 nodes, 32 features, 3 classes
```

Functional checks:

- all four paired models trained successfully
- `full` triggered ASR: 1.0000
- `trigger_only` triggered ASR: 0.0000
- shared-initialization graph invariants passed
- repeated `full` training produced exactly identical parameters and logits on CPU
- node-level CSV, detection metrics, permutation tests, plots, and summary were generated
- aggregate script completed successfully

Diagnostic result:

The synthetic smoke graph produced a perfect target-logit DID signal but did not produce a positive scale-invariant spectral-shape DID signal. This result was retained rather than tuned away. The smoke test validates implementation and controls, not the Cora research hypothesis.

## Platform note

The formal Cora configuration requires CUDA. Windows PowerShell execution has not been run inside this Linux/CPU validation environment; the formal runner includes an explicit CUDA preflight and uses the established `cmd.exe` output-stream wrapper.
