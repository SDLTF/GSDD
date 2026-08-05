# GSDD-v0.6.3

Graph Spectral Discrepancy Defense — **Clean-label Factorial Audit**.

Round 6.2 confirmed that the previous dirty-label validity gate cannot be reused for clean-label attacks: when labels are unchanged, the old `full` and `trigger-only` training conditions are identical. v0.6.3 replaces that invalid comparison with a model × trigger factorial design.

## Core factorial design

Two models are trained from the same initialization and RNG seed:

- $M_c$: trained on the clean graph
- $M_p$: trained on the clean-label poisoned graph

Both models are evaluated under three test-trigger conditions:

- no test trigger, $\varnothing$
- matched context-conditioned trigger, $T$
- victim-shuffled trigger, $T_{\mathrm{shuffle}}$

The shuffled control cyclically reassigns complete generated triggers across victims. It preserves:

- the exact trigger-feature multiset
- trigger size
- trigger topology
- target class

but breaks the learned trigger–victim contextual pairing.

The functional binding score is

$$
\Delta_{\mathrm{CL}}
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

The spectral interaction is

$$
D_{\mathrm{CL}}
=
[S(M_p,T)-S(M_c,T)]
-
[S(M_p,T_{\mathrm{shuffle}})-S(M_c,T_{\mathrm{shuffle}})]
$$

## What the experiment does

1. Trains one clean Cora model and scans all target classes
2. Selects at most two target classes whose natural non-target ASR is at most $0.15$
3. Runs clean-label DPGBA-style pilots with poison counts 4, 8, and 12
4. Applies the clean-label factorial validity gate
5. Expands only a valid pilot to seeds 2026 and 3407
6. Computes matched-vs-shuffled spectral interaction scores
7. Packages the aggregate and all per-run outputs

Poison count is capped at 12 because Cora has only 20 labeled training nodes per class. This preserves at least eight same-class non-victim controls for node-level detection.

## CUDA requirement

Formal Cora experiments require a CUDA-enabled PyTorch installation. There is no silent CPU fallback.

```powershell
python check_cuda_required.py
```

## Run order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v063.ps1
```

The smoke test uses a synthetic graph and CPU only. It validates code paths, not the research hypothesis.

### 2. Recommended full experiment

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

### 3. Pilot only

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_clean_label_factorial_pilot.ps1
```

## Upload after completion

Upload this first:

```text
artifacts\clean_label_factorial_aggregate.zip
```

The larger per-run archive is:

```text
artifacts\gsdd_v063_clean_label_factorial_runs.zip
```

## Formal clean-label admission gate

A run is valid only when all conditions hold:

- $\operatorname{ASR}(M_c,\varnothing)\leq0.10$
- $\operatorname{ASR}(M_c,T)\leq0.20$
- $\operatorname{ASR}(M_p,\varnothing)\leq0.20$
- $\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})\leq0.20$
- $\operatorname{ASR}(M_p,T)\geq0.80$
- $\Delta_{\mathrm{CL}}\geq0.40$

Detection metrics from invalid attacks are retained for diagnosis but are not evidence of defense effectiveness.

Read [`docs/ROUND63_PROTOCOL.md`](docs/ROUND63_PROTOCOL.md) before interpreting the results.
