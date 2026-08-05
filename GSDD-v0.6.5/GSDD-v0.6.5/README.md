# GSDD-v0.6.5

Graph Spectral Discrepancy Defense — **Generic Selective-activation Repair**.

The v0.6.4 pilot produced a nearly valid generic clean-label attack:

$$
\operatorname{ASR}_{\mathrm{poison,trigger}}=0.82
$$

$$
\operatorname{ASR}_{\mathrm{clean,trigger}}=0.31
$$

The attack failed only because the trigger itself already induced the target class on the clean model. v0.6.5 therefore fixes `target=5`, `poison_count=12`, and concentrates on reducing clean-model trigger activation without sacrificing poisoned-model ASR.

## Main methodological change

Clean-label training victims already belong to the target class. Penalizing target probability on those nodes is ill-posed. v0.6.5 computes the new selective-activation losses only on held-out **non-target calibration nodes**.

For matched and shuffled generic triggers, define

$$
p_p(v,T)=P_{M_p}(y_t\mid v\oplus T)
$$

$$
p_c(v,T)=\max\left\{P_{M_c}(y_t\mid v\oplus T),P_{M_l}(y_t\mid v\oplus T)\right\}
$$

The clean probability cap is

$$
\mathcal L_{\mathrm{cap}}
=
\frac12\sum_{Q\in\{T,T_{\mathrm{shuffle}}\}}
\mathbb E_{v\in C}
\left[\max\left(0,p_c(v,Q)-\tau_c\right)^2\right]
$$

The selective-activation loss is

$$
\mathcal L_{\mathrm{sel}}
=
\frac12\sum_{Q\in\{T,T_{\mathrm{shuffle}}\}}
\mathbb E_{v\in C}
\left[\max\left(0,m_s-p_p(v,Q)+p_c(v,Q)\right)\right]
$$

A target-prototype excess penalty discourages the generator from simply copying normal target-class semantics.

## Pilot candidates

All candidates use target class 5 and 12 clean-label poisoned training nodes.

- balanced selective activation
- strong clean-model cap
- subtle low-target-prototype trigger
- attack-preserving high outer-round setting
- compact two-node trigger

Only a pilot satisfying all functional conditions is expanded to seeds 2026 and 3407:

$$
\operatorname{ASR}_{\mathrm{full}}\geq0.80
$$

$$
\operatorname{ASR}_{\mathrm{control,max}}\leq0.20
$$

$$
\Delta_{\mathrm{generic}}\geq0.40
$$

## CUDA requirement

Formal Cora experiments require CUDA-enabled PyTorch. There is no silent CPU fallback.

```powershell
python check_cuda_required.py
```

## Run order

### 1. Smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v065.ps1
```

### 2. Full pilot and automatic multiseed expansion

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

### 3. Pilot only

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_generic_selective_repair_pilot.ps1
```

## Upload after completion

Upload this first:

```text
artifacts\generic_selective_repair_aggregate.zip
```

The full per-run archive is:

```text
artifacts\gsdd_v065_generic_selective_runs.zip
```

See [`docs/ROUND65_PROTOCOL.md`](docs/ROUND65_PROTOCOL.md) for the full protocol.
