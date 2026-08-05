# GSDD-v0.2 Round-2 protocol

## Research question

Does the graph spectral signal observed in the v0.1 Cora run remain detectable when target-class contamination is reduced, and can a label-free or trimmed calibration avoid the median-reference reversal?

## Primary experiment

- Dataset: Cora
- Attack: current structure-plus-feature trigger
- Selection: dirty label
- Poisoned training victims: 4
- Target class: 0
- Supervised model: two-layer GCN
- Label-free reference: DGI
- Bands: four Bernstein graph-frequency bands

## Primary acceptance conditions

The run is interpretable only if:

1. triggered ASR is materially above the clean target-class prediction rate
2. clean test accuracy does not collapse
3. the raw feature report contains at least one stable spectral coordinate

The calibration hypothesis receives preliminary support if either `global_mad_combined`, `global_ecdf_combined`, `trimmed_combined`, or `hybrid_combined` improves materially over `legacy_combined` without selecting score direction from poison labels.

## Secondary contamination sweep

Run poison counts 4, 8, 12, and 20. Plot AUROC and AUPRC against the maximum observed-label contamination. The expected pattern is that legacy calibration degrades near its 50% breakdown point, while label-free calibration should degrade more gradually.

## Clean-label branch

Run four clean-label poisoned target-class nodes. This tests whether frequency signatures arise from the trigger itself when label semantic drift is absent.

## Scientific cautions

- `raw_feature_metrics.csv` is diagnostic-only and uses ground truth to orient each coordinate
- no method should be selected solely from the same run used for evaluation
- a positive Cora result must later be tested on other attacks, seeds, and datasets
