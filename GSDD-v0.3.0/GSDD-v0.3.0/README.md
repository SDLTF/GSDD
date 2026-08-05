# GSDD-v0.3.0

GSDD is a diagnostic prototype for studying graph-spectral signals in GNN backdoor learning.

v0.3 addresses a specific theoretical risk found in the v0.2 Cora result: raw supervised-minus-self-supervised band gain can be dominated by arbitrary hidden-representation scale.

## Recommended experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

This runs Cora with four dirty-label poisoned nodes for seeds 1027, 2026 and 3407, then creates:

```text
artifacts\gsdd_v03_scale_audit_multiseed.zip
artifacts\gsdd_v03_scale_audit_aggregate.zip
```

Upload the first ZIP for full analysis. The second ZIP contains compact aggregate tables.

## Single-seed check

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_scale_audit_single.ps1
```

## New v0.3 quantities

For every layer, v0.3 separates:

- `gain_level`: mean cross-model log gain across bands, which is scale-sensitive
- `centered_delta_gain`: bandwise gain after subtracting the mean, which is scale-invariant
- `gain_shape_norm`: magnitude of the centered transfer profile

A genuine graph-spectral transfer claim requires the centered shape signal to remain detectable across seeds.

See `docs/ROUND3_PROTOCOL.md` for the experiment logic.
