# GSDD-v0.6.4

Graph Spectral Discrepancy Defense — **Dual Clean-label Audit**.

v0.6.3 showed that the current DPGBA-style adapter did not learn victim-specific trigger binding: shuffled triggers were consistently as strong as, or stronger than, matched triggers. v0.6.4 therefore separates two different research hypotheses instead of forcing them through one validity rule.

## Two attack modes

### 1. Generic clean-label trigger

The trigger is intended to transfer across victims. Matched and shuffled pairings should both activate the poisoned model.

The admission score is

$$
\Delta_{\mathrm{generic}}
=
\min\left\{
\operatorname{ASR}(M_p,T),
\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})
\right\}
-
\max\left\{
\operatorname{ASR}(M_c,T),
\operatorname{ASR}(M_c,T_{\mathrm{shuffle}}),
\operatorname{ASR}(M_p,\varnothing),
\operatorname{ASR}(M_c,\varnothing)
\right\}
$$

The generator is trained on both its matched and shuffled outputs and receives a cross-victim consistency regularizer.

### 2. Contextual clean-label trigger

The trigger is intended to depend on its victim context. A matched trigger must outperform a trigger generated for another victim.

The admission score is

$$
\Delta_{\mathrm{context}}
=
\operatorname{ASR}(M_p,T)
-
\max\left\{
\operatorname{ASR}(M_c,T),
\operatorname{ASR}(M_p,T_{\mathrm{shuffle}}),
\operatorname{ASR}(M_p,\varnothing),
\operatorname{ASR}(M_c,\varnothing)
\right\}
$$

The generator receives:

- a matched-vs-shuffled target-logit margin loss
- a context-relation preservation loss
- a minimum trigger-diversity regularizer

## Spectral diagnostics

The two attack modes also use different spectral interactions.

Generic mode measures the trigger-vs-no-trigger model interaction:

$$
D_{\mathrm{generic}}
=
\frac12\sum_{Q\in\{T,T_{\mathrm{shuffle}}\}}
\left[S(M_p,Q)-S(M_c,Q)\right]
-
\left[S(M_p,\varnothing)-S(M_c,\varnothing)\right]
$$

Contextual mode retains the matched-vs-shuffled interaction:

$$
D_{\mathrm{context}}
=
\left[S(M_p,T)-S(M_c,T)\right]
-
\left[S(M_p,T_{\mathrm{shuffle}})-S(M_c,T_{\mathrm{shuffle}})\right]
$$

Detection metrics count as defense evidence only after the corresponding attack passes its own functional gate.

## Candidate search

The pilot automatically selects one low-baseline target class and runs six candidates:

- generic: poison 8, trigger size 3
- generic: poison 12, trigger size 3
- generic: poison 12, trigger size 4
- contextual: poison 8, trigger size 3, pair weight 4
- contextual: poison 12, trigger size 3, pair weight 6
- contextual: poison 12, trigger size 3, pair weight 10

The best valid candidate from each mode is expanded to seeds 2026 and 3407. Invalid modes are not expanded.

## CUDA requirement

Formal Cora experiments require a CUDA-enabled PyTorch installation. There is no silent CPU fallback.

```powershell
python check_cuda_required.py
```

## Run order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v064.ps1
```

### 2. Full experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

### 3. Pilot only

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_dual_clean_label_pilot.ps1
```

## Upload after completion

Upload this first:

```text
artifacts\dual_clean_label_aggregate.zip
```

The larger per-run archive is:

```text
artifacts\gsdd_v064_dual_clean_label_runs.zip
```

Read [`docs/ROUND64_PROTOCOL.md`](docs/ROUND64_PROTOCOL.md) before interpreting the results.
