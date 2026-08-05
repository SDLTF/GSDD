# Validation record

## Static validation

- Python modules compiled successfully with `python -m compileall`
- Configuration files loaded successfully
- All v0.2 diagnostic outputs are generated from the smoke configuration
- PowerShell scripts were scanned for the previous `$variable:` interpolation bug; only intentional `$env:` scoped variables remain

## Runtime smoke validation

Configuration: `configs/smoke.yaml`

Observed result:

- status: success
- clean accuracy: 1.0000
- triggered ASR: 1.0000
- maximum observed-label contamination: 0.2857
- legacy combined AUROC: 0.6368
- global MAD combined AUROC: 0.6731
- global ECDF combined AUROC: 0.4658
- trimmed combined AUROC: 0.6239
- hybrid combined AUROC: 0.5043

The smoke result is retained under `example_smoke_result/`. These numbers only validate the pipeline; they are not evidence for Cora performance.

## Platform limitation

The Python pipeline was executed in the build environment. Windows PowerShell 5.1 is not installed in that environment, so final `.ps1` execution must be verified on the user's Windows machine. The scripts retain the `cmd.exe` output-merging workaround introduced in v0.1.2.
