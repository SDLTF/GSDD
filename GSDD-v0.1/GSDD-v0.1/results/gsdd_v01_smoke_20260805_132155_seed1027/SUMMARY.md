# GSDD-v0.1 Diagnostic Summary

## Run status

- Status: `success`
- Dataset: `synthetic`
- Seed: `1027`
- Device: `cpu`
- Poisoned training victims: `6`
- Clean test accuracy: `1.0000`
- Triggered test ASR: `1.0000`

## Detection metrics

| Score | AUROC | AUPRC | FPR@95TPR | F1 (oracle-k) |
|---|---:|---:|---:|---:|
| structure | 0.4017 | 0.1695 | 1.0000 | 0.1667 |
| input_spectrum | 0.6026 | 0.3034 | 1.0000 | 0.5000 |
| model_js | 0.8376 | 0.4144 | 0.2821 | 0.3333 |
| spectral_relation | 0.5556 | 0.2397 | 0.7949 | 0.3333 |
| transfer | 0.3034 | 0.1082 | 0.9487 | 0.0000 |
| combined | 0.6325 | 0.3195 | 0.7949 | 0.3333 |

## Interpretation

- `model_js` tests pointwise H3: whether supervised and label-free encoders show different band distributions on poisoned nodes.
- `spectral_relation` is the DShield-style same-label distance-contraction version of H3.
- `transfer` tests H4: whether band-wise amplification/suppression gains differ abnormally.
- `combined` is a transparent diagnostic fusion score, not yet a finalized defense objective.
- A high ASR is required before weak detection scores can be interpreted as a failure of the spectral hypothesis.

See `node_scores.csv`, `detection_metrics.json`, and the generated plots for detailed analysis.