# GSDD-v0.1 Diagnostic Summary

## Run status

- Status: `success`
- Dataset: `Cora`
- Seed: `2026`
- Device: `cpu`
- Poisoned training victims: `20`
- Clean test accuracy: `0.7250`
- Triggered test ASR: `1.0000`

## Detection metrics

| Score | AUROC | AUPRC | FPR@95TPR | F1 (oracle-k) |
|---|---:|---:|---:|---:|
| structure | 0.5246 | 0.1477 | 0.8167 | 0.0500 |
| input_spectrum | 0.1654 | 0.0873 | 0.9833 | 0.0000 |
| model_js | 0.3079 | 0.1033 | 0.9750 | 0.0500 |
| spectral_relation | 0.3467 | 0.1136 | 0.9333 | 0.0500 |
| transfer | 0.1475 | 0.0861 | 0.9917 | 0.0000 |
| combined | 0.2846 | 0.1019 | 0.9833 | 0.0500 |

## Interpretation

- `model_js` tests pointwise H3: whether supervised and label-free encoders show different band distributions on poisoned nodes.
- `spectral_relation` is the DShield-style same-label distance-contraction version of H3.
- `transfer` tests H4: whether band-wise amplification/suppression gains differ abnormally.
- `combined` is a transparent diagnostic fusion score, not yet a finalized defense objective.
- A high ASR is required before weak detection scores can be interpreted as a failure of the spectral hypothesis.

See `node_scores.csv`, `detection_metrics.json`, and the generated plots for detailed analysis.