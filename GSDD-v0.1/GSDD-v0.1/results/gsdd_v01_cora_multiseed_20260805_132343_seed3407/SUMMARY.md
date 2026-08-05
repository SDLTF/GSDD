# GSDD-v0.1 Diagnostic Summary

## Run status

- Status: `success`
- Dataset: `Cora`
- Seed: `3407`
- Device: `cpu`
- Poisoned training victims: `20`
- Clean test accuracy: `0.7680`
- Triggered test ASR: `1.0000`

## Detection metrics

| Score | AUROC | AUPRC | FPR@95TPR | F1 (oracle-k) |
|---|---:|---:|---:|---:|
| structure | 0.5171 | 0.1763 | 0.9333 | 0.2000 |
| input_spectrum | 0.3021 | 0.1059 | 0.9500 | 0.0500 |
| model_js | 0.4183 | 0.1232 | 0.9250 | 0.0500 |
| spectral_relation | 0.3592 | 0.1093 | 0.9833 | 0.0000 |
| transfer | 0.3187 | 0.1180 | 0.9417 | 0.0500 |
| combined | 0.3708 | 0.1330 | 0.9833 | 0.0500 |

## Interpretation

- `model_js` tests pointwise H3: whether supervised and label-free encoders show different band distributions on poisoned nodes.
- `spectral_relation` is the DShield-style same-label distance-contraction version of H3.
- `transfer` tests H4: whether band-wise amplification/suppression gains differ abnormally.
- `combined` is a transparent diagnostic fusion score, not yet a finalized defense objective.
- A high ASR is required before weak detection scores can be interpreted as a failure of the spectral hypothesis.

See `node_scores.csv`, `detection_metrics.json`, and the generated plots for detailed analysis.