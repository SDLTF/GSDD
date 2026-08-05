# Output Guide

## `summary.json`

Machine-readable run summary containing:

- run status
- environment and dataset metadata
- attack settings
- clean accuracy
- triggered ASR
- detection metrics

## `SUMMARY.md`

Human-readable compact summary.

## `node_scores.csv`

One row per original labeled training node.

Important columns:

- `node_id`
- `observed_label`
- `is_poisoned`
- `score_structure`
- `score_input_spectrum`
- `score_model_js`
- `score_spectral_relation`
- `score_transfer`
- `score_combined`
- `input_band_*`
- `model_js_layer_*`
- `spectral_relation_layer_*`
- `delta_gain_l*_b*`

## `detection_metrics.json`

For each score:

- AUROC
- AUPRC
- FPR at 95% TPR
- precision, recall, and F1 when selecting a fixed fraction
- precision, recall, and F1 when selecting the true poison count as oracle-k

Oracle-k is diagnostic only. It is not a deployable thresholding method.

## `band_transfer_layer*.png`

Shows the average quantity

$$
T_s-T_u
$$

for clean and poisoned nodes in each frequency band.

A useful plot shows a systematic poisoned-node deviation in one or more bands rather than random oscillation in every band.

## `distribution_transfer.png`

This is the most direct H4 visualization.

## `roc_curves.png` and `pr_curves.png`

These compare all component scores and the combined score.

AUPRC is particularly important because poison-node detection is imbalanced.

## `history_supervised.csv`

Used to verify that the supervised model converged and did not collapse.

## `history_ssl.csv`

Used to verify that the DGI objective decreased.
