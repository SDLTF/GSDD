# Round 3: scale-confound audit

## Motivation

The v0.2 Cora low-poison run produced a strong operational transfer detector, but every raw band gain moved in nearly the same direction. This can arise from an arbitrary multiplicative rescaling of hidden representations rather than a frequency-selective transfer mechanism.

For layer `l` and band `b`, v0.2 used

$$
\Delta T_{l,b}(v)=T^{\mathrm{sup}}_{l,b}(v)-T^{\mathrm{ssl}}_{l,b}(v)
$$

v0.3 decomposes this vector into

$$
\mu_l(v)=\frac{1}{B}\sum_{b=1}^B\Delta T_{l,b}(v)
$$

and

$$
\widetilde{\Delta T}_{l,b}(v)=\Delta T_{l,b}(v)-\mu_l(v)
$$

The level `mu` is scale-sensitive. The centered shape is invariant when the whole hidden representation is multiplied by a positive scalar.

## Decision rule

- If `transfer_level` is strong but `transfer_shape` is near chance, the apparent H4 signal is mainly representation scale
- If `transfer_shape` is stable across seeds, there is evidence for genuine frequency-selective transfer discrepancy
- `global_ecdf_scale_invariant` and `hybrid_scale_invariant` exclude the raw level signal

## Experiment

Run three seeds on Cora with four dirty-label poisoned training nodes

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

Upload

```text
artifacts\gsdd_v03_scale_audit_multiseed.zip
```

The aggregate tables are also packaged in

```text
artifacts\gsdd_v03_scale_audit_aggregate.zip
```

## Primary outputs

- `gain_level_l*.csv` columns in `node_scores.csv`
- `centered_delta_gain_l*_b*` columns
- `gain_shape_norm_l*` columns
- `band_transfer_shape_layer*.png`
- `global_ecdf_transfer_level`
- `global_ecdf_transfer_shape`
- `global_ecdf_transfer_shape_norm`
- `global_ecdf_scale_invariant`
- `hybrid_scale_invariant`
