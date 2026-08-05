# Changelog

## v0.5.0

- Added a single-run four-model paired design with shared initialization and shared training RNG
- Added backdoor-specific spectral difference-in-differences:
  - `did_raw_l2`
  - `did_level_l2`
  - `did_shape_l2`
  - `did_distribution_l2`
- Added non-spectral target-logit and target-probability DID positive controls
- Added victim-set permutation tests rather than relying only on AUROC
- Added a repeated `full` training control to quantify CPU/CUDA numerical nondeterminism
- Added paired graph invariant checks and initialization hashing
- Added three-seed aggregation with duplicate-run handling
- Formal Cora configuration now requests `device: cuda`; the PowerShell runner fails early when CUDA is unavailable
- Added automatic full-result and aggregate ZIP packaging

## Why this round exists

v0.4 showed that prior spectral scores were primarily trigger-anomaly detectors: `trigger_only` and `full` produced similar scores even though only `full` had high ASR. v0.5 holds the trigger graph fixed and subtracts the corresponding clean-graph label effect, isolating the interaction between trigger presence and target-label binding.
