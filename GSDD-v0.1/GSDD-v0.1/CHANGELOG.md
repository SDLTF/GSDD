# Changelog

## 0.1.1 — 2026-08-05

- Fixed Windows PowerShell 5.1 treating Python warnings on stderr as terminating `NativeCommandError` records
- Added a shared `Invoke-GsddCommand` wrapper with reliable logging and exit-code checks
- Updated smoke, Cora, fast Cora, multi-seed, environment, and aggregation scripts
- Enabled explicit sparse COO invariant checks to silence PyTorch's implicit-check warning

## 0.1.0 — 2026-08-05

- Added standalone sparse GCN and DGI implementations
- Added automatic Cora Planetoid loading and a synthetic smoke dataset
- Added controlled structural-feature backdoor injection
- Added Bernstein graph-frequency filter bank
- Added Hutchinson local spectral-moment estimation
- Added pointwise model-JS discrepancy
- Added DShield-style same-label spectral-relation discrepancy
- Added cross-model spectral-transfer discrepancy
- Added robust class- and degree-conditioned calibration
- Added AUROC, AUPRC, FPR@95TPR, top-k metrics, plots, and multi-seed collection
- Added PowerShell setup and run scripts for Windows
- Added a verified end-to-end smoke result without suppressing the negative H4 outcome
