# Validation

Validation date: 2026-08-05

## Completed checks

- all Python files pass `compileall`
- formal runner rejects non-CUDA devices in both CLI override and runtime checks
- PowerShell formal scripts run `check_cuda_required.py` before experiments
- binding-aware attack module builds all three attack families
- four paired graphs preserve the required input and label invariants
- internal CPU-only synthetic unit test completed for both learned families
- aggregate script was syntax checked

## Internal synthetic diagnostic

The internal diagnostic bypassed the formal CUDA-only entry point solely to exercise model and graph logic in this build environment.

For the UGBA-style binding-aware adapter, the diagnostic produced approximately:

- none ASR: 0.00
- label-only ASR: 0.05
- trigger-only ASR: 0.00
- full ASR: 0.70

This shows that the direct-evasion control failure was substantially reduced. The tiny synthetic configuration is not treated as a research result and is intentionally weaker than the Cora configuration.

The DPGBA-style adapter achieved strong local-distribution preservation in the synthetic diagnostic but remained too weak to form a high-ASR backdoor. The formal experiment therefore retains the hard validity gate; failure is reported as `invalid_attack`, not hidden or converted into a GSDD result.

## Environment limitation

The artifact-building container has no CUDA GPU. Formal Windows/CUDA execution must be validated on the user's machine. No code path in the formal experiment silently falls back to CPU.
