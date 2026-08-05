# Round 6.1 protocol: binding-aware attack-validity repair

## Research question

Can an adaptive or distribution-preserving trigger be trained as a functional backdoor rather than a direct targeted evasion perturbation, while retaining the four-model paired DID controls?

## Controlled models

For every admitted attack plan, train from the same initialization and training seed:

- `none`
- `label_only`
- `trigger_only`
- `full`

All ASR values are measured on one shared triggered inference graph.

## Binding-aware generator

A clean surrogate and label-only surrogate share their initialization with a poisoned surrogate. Generator optimization rewards target activation on the poisoned surrogate and preserves original-label predictions on both controls.

The interaction margin is

$$
M_i
=
(z_{i,t}^{P}-z_{i,y_i}^{P})
-
\max\left\{
 z_{i,t}^{C}-z_{i,y_i}^{C},
 z_{i,t}^{L}-z_{i,y_i}^{L}
\right\}
$$

The binding penalty is

$$
\mathcal L_{\mathrm{gap}}
=
\frac1m\sum_i\max(0,\gamma-M_i)
$$

The full generator loss is

$$
\mathcal L_{\mathrm{gen}}
=
\lambda_t\mathcal L_{\mathrm{target}}^{P}
+
\lambda_g\mathcal L_{\mathrm{gap}}
+
\lambda_p\mathcal L_{\mathrm{preserve}}^{C,L}
+
\lambda_e\mathcal L_{\mathrm{evasion}}^{C,L}
+
\mathcal L_{\mathrm{stealth}}
$$

The trigger generator is alternated with refitting the poisoned surrogate. Topology is selected using the poisoned-versus-control probability gap, not poisoned target probability alone.

## Functional admission

The default thresholds are:

- full ASR at least 0.80
- every control ASR at most 0.10
- full minus trigger-only ASR at least 0.60

Only admitted runs count toward Graph Spectral Discrepancy generalization. A high spectral AUROC from an invalid attack must not be presented as a backdoor result.

## Experiment grid

- dataset: Cora
- target class: 0
- poison count: 4
- seeds: 1027, 2026, 3407
- families: fixed baseline, UGBA-style binding-aware, DPGBA-style binding-aware
- device: CUDA only

Learned-family expansion is pilot-gated at seed 1027.

## Decision table

| Outcome | Interpretation |
|---|---|
| Full high, controls low, spectral DID high | attack repair and GSDD generalization both supported |
| Full high, controls low, spectral DID low | valid attack, but spectral mechanism does not generalize |
| Full high, any control high | direct evasion or label-only confounding; invalid attack |
| Full low, controls low | trigger is too weak; attack generator needs revision |
