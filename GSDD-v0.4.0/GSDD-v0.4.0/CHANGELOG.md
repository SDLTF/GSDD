# Changelog

## 0.4.0

- Added a four-condition causal ablation: `none`, `label_only`, `trigger_only`, and `full`
- Added `attack.ablation_mode` and matching CLI override
- Added label-free multivariate `shape_geometry_mahalanobis`
- Projected centered transfer profiles into the zero-sum spectral-shape subspace
- Added multi-seed causal-effect aggregation and factorial interaction estimates
- Added automatic archives for the full result set and compact aggregate tables
- Retained all v0.3 scale-confound diagnostics for direct comparison

## 0.3.0

- Separated cross-model transfer into scale-sensitive level and scale-invariant shape
- Added three-seed scale audit
