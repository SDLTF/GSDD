# Round 6.5 Protocol: Generic Selective-activation Repair

## Research question

Can a generic clean-label graph trigger retain high activation on a poisoned GNN while becoming non-activating on a clean GNN?

The previous best candidate had high poisoned-model ASR but also excessive clean-model trigger ASR. This round isolates that single failure mode.

## Fixed attack setting

- Dataset: Cora
- Target class: 5
- Selection: clean-label
- Poison count: 12
- Base trigger size: 3, with one compact size-2 candidate
- Attack family: DPGBA-style prototype-mixture adapter

## Selective-activation objective

Let $C$ denote held-out non-target calibration nodes. For both matched and victim-shuffled triggers, optimize:

1. poisoned-model target classification
2. clean-model target-probability cap
3. poisoned-minus-clean selectivity margin
4. cross-victim trigger consistency
5. local-context plausibility
6. target-prototype excess suppression

The cap and selectivity terms are deliberately not evaluated on target-class training victims.

## Candidate sweep

| Candidate | Trigger | Cap weight | Cap | Selectivity weight | Margin | Raw blend | Target prototype fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 3 | 16 | 0.15 | 10 | 0.40 | 0.60 | 0.25 |
| clean_strong | 3 | 24 | 0.12 | 14 | 0.45 | 0.55 | 0.20 |
| subtle | 3 | 20 | 0.12 | 12 | 0.45 | 0.48 | 0.10 |
| attack_preserve | 3 | 18 | 0.15 | 12 | 0.45 | 0.58 | 0.20 |
| compact | 2 | 20 | 0.13 | 12 | 0.42 | 0.58 | 0.20 |

## Functional admission gate

Define

$$
\Delta_{\mathrm{generic}}
=
\min\left\{\operatorname{ASR}(M_p,T),\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})\right\}
-
\max\left\{\operatorname{ASR}(M_c,T),\operatorname{ASR}(M_c,T_{\mathrm{shuffle}}),\operatorname{ASR}(M_p,\varnothing),\operatorname{ASR}(M_c,\varnothing)\right\}
$$

A run is valid only when

$$
\operatorname{ASR}_{\mathrm{full}}\geq0.80
$$

$$
\operatorname{ASR}_{\mathrm{control,max}}\leq0.20
$$

$$
\Delta_{\mathrm{generic}}\geq0.40
$$

Detection metrics are interpreted as backdoor-defense evidence only for admitted attacks.

## Outputs

The aggregate includes:

- every candidate's six factorial ASRs
- clean accuracy
- functional validity reasons
- selected candidate and admission distance
- Spectral Hybrid AUROC, AUPRC and FPR@95TPR
- permutation-test result

## Decision rule

- If a pilot passes, expand exactly that configuration to seeds 2026 and 3407
- If no pilot passes, do not expand and use the aggregate to identify whether attack strength or clean selectivity remains limiting
