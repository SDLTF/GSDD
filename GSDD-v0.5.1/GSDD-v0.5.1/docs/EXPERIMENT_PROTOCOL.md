# GSDD-v0.1 Experiment Protocol

## Phase A — Code integrity

Run `configs/smoke.yaml` on the synthetic dataset.

Acceptance conditions:

- Process exits with code 0
- `summary.json` has `status=success`
- `node_scores.csv` exists and contains both clean and poisoned rows
- All five score columns contain finite values
- AUROC/AUPRC and plots are produced

This phase does not assess scientific validity.

## Phase B — Attack validity on Cora

Run `configs/cora_fast.yaml` first.

Check:

- Clean test accuracy is non-degenerate
- Triggered test ASR is materially above the target-class base rate
- Supervised training converges
- DGI loss decreases

If ASR is weak, adjust only the controlled attack parameters before interpreting H3/H4:

- increase `poison_count`
- increase `trigger_feature_count`
- retain `stamp_victim_features: true`
- increase supervised epochs

Do not tune spectral detection parameters against poison labels before attack validity is established.

## Phase C — Hypothesis test

Run the standard config for seeds:

$$
1027,2026,3407
$$

Primary endpoints:

$$
\operatorname{AUROC}(S_{\mathrm{transfer}})
$$

$$
\operatorname{AUPRC}(S_{\mathrm{transfer}})
$$

Secondary endpoints:

$$
\operatorname{AUROC}(S_{\mathrm{model}})
$$

$$
\operatorname{AUROC}(S_A)
$$

$$
\operatorname{AUROC}(S_X)
$$

The core claim is supported only when transfer or model discrepancy adds information beyond structural and raw-input anomaly scores.

## Phase D — Necessary ablations for v0.1 report

1. Structure only
2. Input spectrum only
3. Model JS only
4. Transfer only
5. Combined
6. One hidden layer versus two hidden layers
7. Two, four, and six Bernstein bands
8. Spectral moments removed
9. DGI replaced with an untrained encoder as a negative control
10. Poison count sensitivity

## Phase E — Transition to v0.2

Only after H3/H4 are supported should the package be extended with:

- official UGBA/GTA/GCBA adapters
- clean-label attacks
- DShield output compatibility
- weighted retraining defense
- heterophilic datasets
- adaptive spectral-preserving attacks
