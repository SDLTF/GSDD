# Validation

Validation date: 2026-08-05

## Completed checks

- all Python source files pass `compileall`
- all YAML configuration files parse successfully
- the v0.6.2 runner exposes target-class, poison-count, and selection-method overrides
- the mixed DPGBA prototype bank handles target and background prototype pools deterministically
- aggregate candidate ranking is syntax checked and tested with synthetic summary fixtures
- PowerShell scripts use the shared warning-safe command wrapper
- formal scripts run the CUDA availability check before experiments
- no formal Python entry point silently falls back to CPU

## Environment limitation

The artifact-building container has no CUDA GPU and cannot execute the formal Cora candidate sweep. Windows/CUDA runtime validation must therefore be completed on the user's machine. The package reports failed or invalid attacks explicitly and does not convert them into defense results.
