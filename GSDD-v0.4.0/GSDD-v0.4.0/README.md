# GSDD-v0.4.0

GSDD-v0.4 tests whether the scale-invariant graph spectral transfer signal found in
v0.3 is genuinely associated with the backdoor trigger, or is instead explained by
dirty-label semantic conflict.

## What changed

The same seeded non-target training nodes are evaluated under a $2\times2$ intervention:

| Mode | Trigger | Dirty-label relabel |
|---|---:|---:|
| `none` | No | No |
| `label_only` | No | Yes |
| `trigger_only` | Yes | No |
| `full` | Yes | Yes |

The detector also adds `shape_geometry_mahalanobis`, a label-free multivariate
distance in the joint zero-sum transfer-shape space. It does not select a band
direction using poison ground truth.

## Recommended run

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

This runs four modes over seeds `1027`, `2026`, and `3407`, then creates:

```text
artifacts\gsdd_v04_causal_ablation_multiseed.zip
artifacts\gsdd_v04_causal_ablation_aggregate.zip
```

Upload the compact aggregate ZIP first. The full multiseed archive is useful when
node-level inspection is required.

## Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v04.ps1
```

## Interpretation

A trigger-dependent spectral mechanism should satisfy all of the following:

1. `none` stays near chance
2. `label_only` does not explain the complete signal
3. `trigger_only` or `full` produces a clear scale-invariant shape shift
4. `full` ASR is high while `none`, `label_only`, and `trigger_only` ASR remain low
5. the result is consistent across seeds

The interaction contrast is

$$
I=S_{\mathrm{full}}-S_{\mathrm{label}}-S_{\mathrm{trigger}}+S_{\mathrm{none}}
$$

Positive $I$ means trigger and label conflict reinforce one another beyond additive
main effects.

See `docs/ROUND4_PROTOCOL.md` for the full decision logic.
