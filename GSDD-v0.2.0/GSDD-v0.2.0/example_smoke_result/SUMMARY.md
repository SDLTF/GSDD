# GSDD-v0.2 Diagnostic Summary

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

## Strongest primitive spectral coordinates

Ground-truth poison labels are used only in this section to audit whether a raw coordinate contains signal. These directions are not used to construct operational scores.

| Group | Feature | Poison direction | Oriented AUROC | Clean mean | Poison mean |
|---|---|---:|---:|---:|---:|
| input_spectrum | input_band_1 | high | 0.9701 | 0.03801 | 0.105976 |
| transfer | delta_gain_l1_b1 | low | 0.9487 | 1.67584 | 1.09295 |
| input_spectrum | input_band_0 | low | 0.8761 | 0.933405 | 0.850019 |
| structure | log_moment_1 | high | 0.8333 | 0.815154 | 0.892711 |
| structure | log_moment_2 | high | 0.8077 | 0.930872 | 1.03195 |
| structure | log_moment_0 | high | 0.8034 | 0.733869 | 0.779695 |
| structure | log_moment_3 | high | 0.7479 | 1.07955 | 1.19574 |
| spectral_relation | spectral_relation_layer_2 | high | 0.7436 | 0.00108007 | 0.00155588 |
| transfer | delta_gain_l2_b1 | low | 0.7265 | 4.67127 | 4.01259 |
| input_spectrum | input_band_2 | high | 0.7094 | 0.0207401 | 0.0345693 |

## Interpretation

- `legacy_combined` reproduces the v0.1 label/degree-conditioned MAD calibration.
- `global_mad_combined` uses no observed labels and therefore cannot be reversed by a majority-poisoned target class.
- `global_ecdf_combined` is a direction-free empirical two-tail score.
- `trimmed_combined` removes globally suspicious nodes before rebuilding class references.
- `hybrid_combined` rank-fuses global MAD, global ECDF, and trimmed calibration.
- `raw_feature_metrics.csv` is a diagnostic signal audit, not a deployable detector, because it uses poison ground truth to select the better direction.

See `node_scores.csv`, `detection_metrics.json`, `raw_feature_metrics.csv`, and the generated comparison plots.