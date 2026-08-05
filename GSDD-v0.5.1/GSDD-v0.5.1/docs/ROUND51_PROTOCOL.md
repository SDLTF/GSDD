# Round 5.1 protocol: deterministic reproducibility audit

## Question

Round 5 found strong backdoor-specific spectral DID signals, but one CUDA repeat control showed a large numerical deviation. Round 5.1 asks whether identical paired-DID experiments preserve their node-level scores and candidate ranking.

## Design

For each seed, run the complete four-model paired experiment twice with identical:

- graph and selected victims
- model initialization
- training seed and dropout sequence
- optimizer and stopping rule
- deterministic-algorithm settings

The internal one-model repeat from v0.5 is disabled because the new audit repeats the entire paired pipeline.

## Primary reproducibility quantities

For each score $s$ and node $v$, let $s_v^{(1)}$ and $s_v^{(2)}$ denote the two replicas. Report:

$$
\rho_{\mathrm P}(s^{(1)},s^{(2)})
$$

$$
\rho_{\mathrm S}(s^{(1)},s^{(2)})
$$

$$
\operatorname{Overlap@k}
=
\frac{|\operatorname{TopK}(s^{(1)})\cap\operatorname{TopK}(s^{(2)})|}{k}
$$

as well as absolute AUROC and AUPRC drift.

The operational candidate set uses the top $5\%$ of labeled training nodes. The oracle victim-count overlap is also retained as a diagnostic, but it is not the deployment criterion.

## Default stability thresholds

- Pearson correlation at least $0.95$
- Spearman correlation at least $0.95$
- operational top-$k$ overlap at least $0.80$
- absolute AUROC drift at most $0.02$
- absolute AUPRC drift at most $0.05$

The primary scale-invariant scores are:

- `did_shape_l2`
- `did_distribution_l2`
- `did_spectral_hybrid`
- `did_shape_mahalanobis`

## Three execution modes

### Practical CUDA audit

Uses deterministic algorithms with `warn_only=true`. This is the main three-seed experiment because some CUDA sparse kernels may not provide a strict deterministic implementation on every PyTorch build. The empirical two-run stability statistics remain the source of truth.

### Strict CUDA audit

Uses `warn_only=false` for seed 3407. If a CUDA sparse operation lacks a deterministic implementation, PyTorch should stop and identify the unsupported operation rather than silently continue.

### Strict CPU reference

Runs seed 3407 twice on CPU with strict deterministic algorithms. It is slower, but provides a reference when strict CUDA execution is unavailable.

## Decision

Proceed to attack and dataset generalization only if the scale-invariant node rankings and operational top-$k$ sets are stable across identical replicas. Exact parameter equality is desirable but not required when downstream scores and decisions remain stable; large score/rank drift is disqualifying even if aggregate AUROC happens to remain similar.
