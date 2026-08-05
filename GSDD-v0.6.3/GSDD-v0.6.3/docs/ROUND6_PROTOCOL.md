# GSDD-v0.6 attack-family generalization protocol

## Research question

GSDD-v0.5 established a backdoor-specific spectral difference-in-differences signal for one fixed synthetic trigger on Cora. Round 6 asks whether that signal survives when the trigger-generation mechanism changes.

For each attack family and random seed, the experiment trains four GCNs with shared initialization and training RNG:

| Model | Trigger graph | Victim labels changed |
|---|---:|---:|
| `none` | no | no |
| `label_only` | no | yes |
| `trigger_only` | yes | no |
| `full` | yes | yes |

The primary interaction remains

$$
D_{\mathrm{DID}}
=
[T_{\mathrm{full}}-T_{\mathrm{trigger-only}}]
-
[T_{\mathrm{label-only}}-T_{\mathrm{none}}]
$$

The principal scale-invariant outputs are `did_shape_l2`, `did_distribution_l2`, `did_shape_mahalanobis`, and `did_spectral_hybrid`.

## Attack families

### `fixed_rare_clique`

The original GSDD mechanism baseline. Each victim receives a three-node clique whose features occupy globally rare coordinates. This trigger is intentionally conspicuous and is retained as a continuity control.

### `ugba_style_adaptive`

A self-contained mechanism-faithful adapter inspired by UGBA:

- a victim-conditioned neural generator creates sparse trigger-node features
- a provisional poisoned surrogate supplies the target-class optimization signal
- a homophily penalty encourages similarity to the victim neighborhood
- clique, star, chain, and cycle motifs are searched using target confidence minus a density penalty

The official UGBA work deliberately selects poisoning nodes and trains an adaptive trigger generator to obtain effective, difficult-to-notice triggers. Official repository: https://github.com/ventr1c/UGBA

### `dpgba_style_distribution`

A self-contained mechanism-faithful adapter inspired by DPGBA:

- trigger features are convex mixtures of real target-class training prototypes
- the optimization penalizes feature mean, variance, and nearest-prototype mismatch
- a neighborhood-homophily term and sparse-topology search are retained

The official DPGBA work extends UGBA from a distribution-preserving perspective and explicitly balances attack performance against outlier-detection stealthiness. Official repository: https://github.com/zzwjames/DPGBA

## Important naming restriction

The two learned families are **not verbatim executions of the official repositories**. They intentionally avoid the old PyTorch Geometric dependency stack and share the exact GSDD graph loader, GCN, split, and paired-control implementation. Results must therefore be reported using the `_style_` names and not as official UGBA or official DPGBA reproduction numbers.

This round is a controlled mechanism-generalization experiment. Exact external-repository reproduction should be a later interoperability round after this common-code benchmark is understood.

## CUDA policy

Round 6 is CUDA-only.

- `configs/cora_attack_generalization.yaml` sets `device: cuda`
- `run_gsdd_v06.py` rejects CPU configurations
- `check_cuda_required.py` fails before the multi-run script starts when CUDA is unavailable
- no CPU fallback exists

## Experimental grid

Attack families:

$$
\texttt{fixed\_rare\_clique},\quad
\texttt{ugba\_style\_adaptive},\quad
\texttt{dpgba\_style\_distribution}
$$

Random seeds:

$$
1027,\quad2026,\quad3407
$$

Total paired runs:

$$
3\text{ families}\times3\text{ seeds}=9
$$

Each run trains four controlled GCNs. The learned families additionally train one provisional poisoned surrogate and one trigger generator.

## Success criteria

A family supports attack-generalized spectral DID only when all of the following hold:

1. `full` ASR is at least $0.80$
2. `trigger_only` ASR is at most $0.30$
3. `none` and `label_only` ASR are at most $0.20$
4. `did_shape_l2` or `did_distribution_l2` has AUROC at least $0.70$
5. the corresponding victim-set permutation test has $p\leq0.05$
6. the conclusion is present across random seeds rather than driven by one run

A high `trigger_only` ASR means the trigger also acts as a direct evasion perturbation. Such a run cannot be interpreted as a pure learned-backdoor mechanism even when detection is strong.

## Commands

CUDA smoke test:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v06.ps1
```

Full nine-run experiment:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

Single family and seed:

```powershell
powershell.exe -ExecutionPolicy Bypass `
    -File .\scripts\run_attack_family_single.ps1 `
    -AttackFamily ugba_style_adaptive `
    -Seed 1027
```

## Output

Upload first:

```text
artifacts\attack_generalization_aggregate.zip
```

If node-level inspection is needed, also upload:

```text
artifacts\gsdd_v06_attack_generalization_multiseed.zip
```
