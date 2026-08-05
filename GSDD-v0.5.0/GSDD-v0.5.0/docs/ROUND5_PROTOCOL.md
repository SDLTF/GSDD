# Round 5 Protocol: Paired Backdoor-Specific Spectral DID

## Research question

The v0.4 factorial ablation showed that most previous spectral scores detect the trigger intervention itself:

- `trigger_only` and `full` had similar spectral anomaly scores
- only `full` produced a functional backdoor

Round 5 therefore asks a narrower question:

> After holding the graph, trigger, initialization, and training randomness fixed, does binding the trigger to the target label create a measurable graph-spectral interaction beyond ordinary dirty-label learning?

## Four controlled models

For every seed, the experiment selects one victim set and constructs four models:

| Model | Trigger graph | Victim relabeling |
|---|---:|---:|
| `none` | no | no |
| `label_only` | no | yes |
| `trigger_only` | yes | no |
| `full` | yes | yes |

The models share exactly the same initial `state_dict`. The RNG is reset to the same training seed immediately before every training run. Therefore:

- `full` and `trigger_only` differ only in labels
- `label_only` and `none` differ only in labels

## Main estimand

Let $T_m^{(l)}(v)\in\mathbb R^B$ denote the layer-$l$ log spectral-gain profile for model $m$ at node $v$.

The trigger-graph label-binding contrast is

$$
B_T^{(l)}(v)=T_{\mathrm{full}}^{(l)}(v)-T_{\mathrm{trigger-only}}^{(l)}(v)
$$

The clean-graph dirty-label contrast is

$$
B_C^{(l)}(v)=T_{\mathrm{label-only}}^{(l)}(v)-T_{\mathrm{none}}^{(l)}(v)
$$

The paired difference-in-differences is

$$
D_{\mathrm{DID}}^{(l)}(v)=B_T^{(l)}(v)-B_C^{(l)}(v)
$$

This removes the trigger input signature and the additive effect of label conflict.

## Scale decomposition

For each layer,

$$
\mu_l(v)=\frac1B\sum_{b=1}^B D_{\mathrm{DID},b}^{(l)}(v)
$$

$$
S_l(v)=D_{\mathrm{DID}}^{(l)}(v)-\mu_l(v)\mathbf 1
$$

The principal scale-invariant statistic is

$$
D_{\mathrm{shape}}(v)=\sqrt{\sum_l\|S_l(v)\|_2^2}
$$

The normalized hidden-frequency distribution also yields

$$
D_{\mathrm{dist}}(v)=\left\|[p_{\mathrm{full}}-p_{\mathrm{trigger-only}}]-[p_{\mathrm{label-only}}-p_{\mathrm{none}}]\right\|_2
$$

## Positive-control anchor

The experiment also measures target-logit DID:

$$
D_{\mathrm{logit}}(v)=\left|[z_{\mathrm{full},t}-z_{\mathrm{trigger-only},t}]-[z_{\mathrm{label-only},t}-z_{\mathrm{none},t}]\right|
$$

This is not a spectral score. It verifies that the paired design captures the learned trigger-label association. A strong logit DID with weak spectral DID means the functional backdoor exists but the proposed spectral mechanism is unsupported.

## Numerical repeat control

The `full` model is trained twice using the same:

- graph and labels
- initial parameters
- RNG seed
- optimizer and hyperparameters

The package reports parameter and logit maximum absolute differences. This quantifies CUDA nondeterminism instead of treating repeated-run variation as a method effect.

## Formal experiment

Datasets and settings:

- Cora
- seeds: 1027, 2026, 3407
- poison count: 4
- trigger size: 3
- four Bernstein frequency bands
- two GCN hidden layers
- CUDA required by the formal PowerShell script

Run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

Outputs:

```text
artifacts\gsdd_v05_paired_did_multiseed.zip
artifacts\gsdd_v05_paired_did_aggregate.zip
```

## Decision criteria

A backdoor-specific spectral mechanism is supported only when:

1. `full` ASR is high, while `none`, `label_only`, and `trigger_only` ASR remain low
2. `did_shape_l2` or `did_distribution_l2` has stable AUROC across seeds
3. its victim-set permutation test is significant
4. the effect is not explained solely by `did_level_l2`
5. repeat-control numerical differences are small relative to the observed effect
