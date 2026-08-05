# Round 4 protocol: causal source of spectral transfer shape

## Research question

Does the scale-invariant transfer-shape signal arise from:

- the graph/feature trigger itself
- dirty-label semantic conflict
- an interaction between trigger and relabeling
- random selected-node variation

## Controlled design

For every seed, `_select_victims` chooses the same four non-target Cora training nodes.
Only the intervention changes.

### None

No graph, feature, or label modification is applied. The selected nodes are marked
only for diagnostic evaluation. AUROC should be near $0.5$.

### Label only

The selected nodes are relabeled to the target class, but no trigger is attached.
A high score here indicates that the diagnostic is responding to semantic label
conflict rather than specifically to a trigger.

### Trigger only

The graph/feature trigger is attached, but labels are preserved. This condition
tests whether the frequency-shape signal responds to the perturbation independently
of a learned target-class backdoor.

### Full

The trigger is attached and dirty-label relabeling is applied. This is the original
backdoor condition and should attain high ASR.

## Primary metrics

Operational, ground-truth-free scores:

- `global_ecdf_transfer_shape_norm`
- `shape_geometry_mahalanobis`
- `global_ecdf_scale_invariant`

Attack behavior:

- clean accuracy
- triggered ASR

Diagnostic-only primitive-coordinate reports remain secondary because their
orientation is chosen using poison ground truth.

## Decision table

| Observation | Interpretation |
|---|---|
| `none` high | detector is not calibrated or victim selection is confounded |
| `label_only` high, `trigger_only` low | mostly semantic/label conflict |
| `trigger_only` high, `label_only` low | primarily trigger-induced spectral change |
| both main effects high | both mechanisms contribute |
| `full` exceeds both strongly | nonlinear trigger-label interaction |
| all conditions near chance | v0.3 signal was unstable |
| shape norm works but geometry fails | radial magnitude matters more than joint direction |
| geometry works across seeds | multivariate spectral-shape cluster is stable |

## Factorial contrast

For a selected-minus-reference score contrast $S$:

$$
T=\frac12(S_{\mathrm{trigger}}+S_{\mathrm{full}}-S_{\mathrm{none}}-S_{\mathrm{label}})
$$

$$
L=\frac12(S_{\mathrm{label}}+S_{\mathrm{full}}-S_{\mathrm{none}}-S_{\mathrm{trigger}})
$$

$$
I=S_{\mathrm{full}}-S_{\mathrm{label}}-S_{\mathrm{trigger}}+S_{\mathrm{none}}
$$

These are reported per seed and then summarized.
