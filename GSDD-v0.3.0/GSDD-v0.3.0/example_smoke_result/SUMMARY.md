# GSDD-v0.3 Scale-Confound Diagnostic Summary

## Run status

- Status: `success`
- Dataset: `synthetic`
- Seed: `1027`
- Device: `cpu`
- Attack mode: `dirty_label`
- Poisoned training victims: `6`
- Clean test accuracy: `1.0000`
- Triggered test ASR: `1.0000`
- Maximum observed-label contamination (diagnostic): `0.2857`

## Calibration comparison

| Score | AUROC | AUPRC | FPR@95TPR | F1 (oracle-k) |
|---|---:|---:|---:|---:|
| legacy_combined | 0.6368 | 0.4148 | 0.8718 | 0.3333 |
| global_mad_combined | 0.6731 | 0.2141 | 0.6410 | 0.1667 |
| global_ecdf_combined | 0.4658 | 0.3341 | 0.9487 | 0.3333 |
| trimmed_combined | 0.6239 | 0.2084 | 0.6154 | 0.3333 |
| hybrid_combined | 0.5043 | 0.2300 | 0.8205 | 0.3333 |
| global_ecdf_scale_invariant | 0.4615 | 0.1414 | 0.8462 | 0.0000 |
| hybrid_scale_invariant | 0.4701 | 0.1681 | 0.8718 | 0.1667 |

## Strongest primitive spectral coordinates

Ground-truth poison labels are used only in this section to audit whether a raw coordinate contains signal. These directions are not used to construct operational scores.

| Group | Feature | Poison direction | Oriented AUROC | Clean mean | Poison mean |
|---|---|---:|---:|---:|---:|
| input_spectrum | input_band_1 | high | 0.9701 | 0.03801 | 0.105976 |
| transfer | delta_gain_l1_b1 | low | 0.9487 | 1.67584 | 1.09295 |
| transfer_shape | centered_delta_gain_l1_b3 | high | 0.9103 | -0.263517 | 0.117422 |
| input_spectrum | input_band_0 | low | 0.8761 | 0.933405 | 0.850019 |
| structure | log_moment_1 | high | 0.8333 | 0.815154 | 0.892711 |
| transfer_level | gain_level_l1 | low | 0.8248 | 1.38361 | 1.16137 |
| structure | log_moment_2 | high | 0.8077 | 0.930872 | 1.03195 |
| structure | log_moment_0 | high | 0.8034 | 0.733869 | 0.779695 |
| transfer_shape | centered_delta_gain_l1_b1 | low | 0.7650 | 0.292225 | -0.0684139 |
| transfer_shape | centered_delta_gain_l2_b3 | high | 0.7607 | -0.73296 | -0.285881 |

## Interpretation

- `legacy_combined` reproduces the v0.1 label/degree-conditioned MAD calibration.
- `global_mad_combined` uses no observed labels and therefore cannot be reversed by a majority-poisoned target class.
- `global_ecdf_combined` is a direction-free empirical two-tail score.
- `trimmed_combined` removes globally suspicious nodes before rebuilding class references.
- `hybrid_combined` rank-fuses global MAD, global ECDF, and trimmed calibration.
- `global_ecdf_scale_invariant` excludes the scale-sensitive raw transfer level and uses centered transfer shape.
- `hybrid_scale_invariant` rank-fuses the scale-invariant component family.
- `raw_feature_metrics.csv` is a diagnostic signal audit, not a deployable detector, because it uses poison ground truth to select the better direction.
- Compare `transfer_level` against `transfer_shape` before claiming a genuine graph-frequency mechanism.

See `node_scores.csv`, `detection_metrics.json`, `raw_feature_metrics.csv`, and the generated comparison plots.