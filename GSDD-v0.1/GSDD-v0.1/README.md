# GSDD-v0.1 Diagnostic Experiment

**Graph Spectral Discrepancy Defense — diagnostic prototype**

This package tests the central hypothesis behind GSDD:

> Poisoned training nodes exhibit an abnormal difference between how a label-free self-supervised encoder and a potentially compromised supervised encoder amplify or suppress graph-frequency bands.

Version 0.1 is a **diagnostic experiment**, not a finalized defense. It answers four questions:

- H1: Are poisoned nodes abnormal in local structural spectral moments?
- H2: Are poisoned nodes abnormal in their input-feature band distribution?
- H3: Do supervised and self-supervised hidden representations have different band distributions on poisoned nodes?
- H4: Is the cross-model band-wise transfer discrepancy larger on poisoned nodes?

The implementation deliberately does **not** depend on PyTorch Geometric. GCN, DGI, sparse normalized adjacency, normalized Laplacian, Bernstein spectral filters, Cora loading, trigger injection, metrics, and plots are implemented directly with PyTorch and SciPy.

## 1. Package contents

```text
GSDD-v0.1/
├─ configs/
│  ├─ smoke.yaml              # fast synthetic test
│  ├─ cora_fast.yaml          # shortened Cora diagnostic
│  └─ default.yaml            # standard Cora diagnostic
├─ docs/
│  ├─ THEORY.md
│  ├─ EXPERIMENT_PROTOCOL.md
│  └─ OUTPUT_GUIDE.md
├─ gsdd/
│  ├─ config.py
│  ├─ data.py                 # Cora loader + controlled backdoor injection
│  ├─ diagnostics.py
│  ├─ graph_ops.py
│  ├─ models.py               # sparse GCN + DGI
│  ├─ spectral.py             # Bernstein filter bank + spectral moments
│  ├─ train.py
│  └─ utils.py
├─ scripts/
│  ├─ setup_env.ps1
│  ├─ run_smoke.ps1
│  ├─ run_cora_fast.ps1
│  ├─ run_cora.ps1
│  ├─ run_multiseed.ps1
│  └─ collect_results.ps1
├─ check_environment.py
├─ collect_results.py
├─ run_gsdd_v01.py
├─ requirements-runtime.txt
└─ requirements.txt
```

## 2. Environment

Recommended:

- Windows 11 or Linux
- Python 3.10–3.12
- PyTorch 2.x
- CUDA-enabled PyTorch when a GPU is available

The code also runs on CPU.

The setup script **does not reinstall PyTorch**, so it will not silently replace an existing CUDA build.

```powershell
cd GSDD-v0.1
powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## 3. First run: synthetic smoke test

Run this before downloading Cora:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke.ps1
```

A successful run ends with output similar to:

```text
[Done] clean_acc=... ASR=...
[H4] transfer AUROC=... AUPRC=...
RESULT_DIR=...\results\gsdd_v01_smoke_..._seed1027
```

The smoke test verifies code paths and output integrity. Its numerical result is not evidence for the research hypothesis.

## 4. Cora diagnostic

Shortened run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_cora_fast.ps1
```

Standard run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_cora.ps1
```

On the first Cora run, the original Planetoid files are downloaded from:

- https://github.com/kimiyoung/planetoid

They are placed under:

```text
data/Planetoid/Cora/raw/
```

If automatic downloading is blocked, manually place these files there:

```text
ind.cora.x
ind.cora.tx
ind.cora.allx
ind.cora.y
ind.cora.ty
ind.cora.ally
ind.cora.graph
ind.cora.test.index
```

## 5. What the controlled attack does

The default experiment uses a controlled node-classification backdoor:

1. Select non-target labeled training nodes
2. Attach a small clique trigger to each selected victim
3. Give all trigger nodes a shared rare-feature signature
4. Stamp the same target-independent shortcut signature onto each victim node
5. Relabel the victim to the target class in dirty-label mode
6. Attach the same trigger to non-target test nodes to measure attack success rate

This is intentionally transparent and reproducible. It is not claimed to reproduce every detail of UGBA, GTA, or GCBA.

The diagnostic should only be interpreted when the attack succeeds. If triggered-test ASR is low, a low detection AUROC does not refute GSDD because the supervised model did not learn a strong backdoor.

## 6. Core spectral quantities

The normalized graph Laplacian is

$$
L=I-D^{-1/2}AD^{-1/2}
$$

GSDD-v0.1 uses a four-band Bernstein filter bank over $L/2$:

$$
g_b(\lambda)
=
\binom{3}{b}
\left(\frac{\lambda}{2}\right)^b
\left(1-\frac{\lambda}{2}\right)^{3-b}
$$

For node $v$ and graph signal $Z$, the band energy is

$$
R_b(v;Z)
=
\left\|e_v^\top g_b(L)Z\right\|_2^2
$$

The normalized band distribution is

$$
p_b(v;Z)
=
\frac{R_b(v;Z)+\varepsilon}
{\sum_cR_c(v;Z)+B\varepsilon}
$$

For supervised and self-supervised hidden states, GSDD computes:

- Jensen–Shannon discrepancy between their band distributions
- Band-wise log gain relative to the input signal
- Supervised gain minus self-supervised gain
- Robust class- and degree-conditioned anomaly scores

See `docs/THEORY.md` for the exact definitions.

## 7. Main outputs

Every run creates a timestamped directory under `results/` containing:

```text
SUMMARY.md
summary.json
node_scores.csv
detection_metrics.json
history_supervised.csv
history_ssl.csv
environment.json
config_resolved.yaml
roc_curves.png
pr_curves.png
distribution_*.png
band_transfer_layer*.png
supervised_gcn.pt
dgi_encoder.pt
```

The most important fields are:

- `triggered_test_asr`: whether the controlled attack was actually learned
- `model_js.auroc`: pointwise H3 diagnostic strength
- `spectral_relation.auroc`: same-label spectral relation contraction
- `transfer.auroc`: H4 diagnostic strength
- `combined.auroc`: diagnostic fusion result
- `clean_test_accuracy`: normal classification utility

## 8. Multi-seed run

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_multiseed.ps1
```

This runs seeds 1027, 2026, and 3407, then writes:

```text
results/aggregate_results.csv
```

Do not draw a conclusion from one seed alone.

## 9. Interpretation rules

A useful first-stage result should satisfy both conditions:

1. The backdoored supervised model has a materially elevated ASR
2. H3 or H4 has detection performance clearly above chance across multiple seeds

Suggested reading:

- ASR low, AUROC low: attack construction or training failed
- ASR high, H3 low, H4 high: transfer discrepancy is more informative than distribution discrepancy
- ASR high, H3 high, H4 high: strong support for the cross-view spectral hypothesis
- ASR high, structure high but H3/H4 low: the method is mostly rediscovering structural trigger anomalies
- ASR high, all scores low: current GSDD hypothesis is not supported under this setting

## 10. Scope and limitations

Version 0.1 does not yet include:

- Official UGBA/GTA/GCBA attack implementations
- DShield baseline code integration
- Clean-label benchmark suite
- Heterophilic datasets
- Adaptive spectrum-preserving attacks
- Final weighted retraining defense
- Formal finite-sample false-positive guarantees

These belong to v0.2 and later. Version 0.1 is designed to answer one question cleanly: **does cross-model graph-frequency transfer discrepancy exist at all?**
