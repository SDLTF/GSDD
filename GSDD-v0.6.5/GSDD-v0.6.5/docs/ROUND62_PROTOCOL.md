# Round 6.2 protocol: Attack Validity Repair Round 2

## Objective

Round 6.1 established that the fixed rare-clique attack is functional, while the learned adapters failed for different reasons:

- UGBA-style reached high full ASR but exceeded the label-only control threshold
- DPGBA-style kept controls low but failed to establish a strong trigger-target mapping

Round 6.2 repairs these two failure modes separately. The hard functional admission gate is unchanged.

## UGBA-style repair

The Round 6.1 pilot used four dirty-label poisoned nodes and obtained approximately

$$
\operatorname{ASR}_{\mathrm{full}}=0.99
$$

$$
\operatorname{ASR}_{\mathrm{label-only}}=0.13
$$

The trigger was functional, but relabeling alone shifted the target prior too strongly. Round 6.2 therefore screens:

- poison count 2, target class 0
- poison count 3, target class 0
- poison count 3, target class 1
- poison count 3, target class 2

The generator remains binding-aware and is given stronger clean and label-only evasion penalties. No admission threshold is relaxed.

## DPGBA-style repair

Round 6.1 dirty-label DPGBA-style obtained only low full ASR. Round 6.2 changes the mechanism to clean-label target-class poisoning:

- selected training victims already belong to the target class
- labels are unchanged
- trigger-bearing target examples teach the trigger-target association
- the label-only intervention becomes identical to the no-attack graph

The generator uses a mixed prototype bank:

$$
\mathcal P
=
\mathcal P_{\mathrm{target}}
\cup
\mathcal P_{\mathrm{background}}
$$

A target-class fraction supplies binding capacity, while background prototypes, neighbor blending, distribution loss, and clean-surrogate penalties constrain direct evasion. Candidate poison counts are 8, 12, and 16.

## Functional admission gate

A run is valid only when

$$
\operatorname{ASR}_{\mathrm{full}}\geq0.80
$$

$$
\max\left\{
\operatorname{ASR}_{\mathrm{none}},
\operatorname{ASR}_{\mathrm{label-only}},
\operatorname{ASR}_{\mathrm{trigger-only}}
\right\}\leq0.10
$$

$$
\operatorname{ASR}_{\mathrm{full}}
-
\operatorname{ASR}_{\mathrm{trigger-only}}
\geq0.60
$$

The pilot candidate grid runs at seed 1027. Only a candidate that passes all three conditions is automatically expanded to seeds 2026 and 3407.

## Primary outputs

The aggregate contains:

- `ATTACK_VALIDITY_REPAIR_ROUND2_SUMMARY.md`
- `repair2_runs.csv`
- `repair2_candidate_ranking.csv`
- `selected_candidates.json`
- `repair2_success_criteria.csv`
- `repair2_group_stats.csv`

Invalid candidates remain in the audit tables but cannot support a GSDD generalization claim.
