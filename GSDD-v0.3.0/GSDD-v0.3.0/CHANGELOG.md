# Changelog

## v0.3.0

- Decomposed raw cross-model log gain into scale-sensitive `transfer_level` and scale-invariant `transfer_shape`
- Added per-layer centered gain coordinates and shape norms
- Added scale-invariant ECDF and hybrid combined scores
- Added centered band-profile plots
- Added three-seed scale-confound audit workflow
- Added aggregate scale-audit tables and Markdown summary
- Retained all v0.2 scores for exact comparison

## v0.2.0

- Added label-free global MAD and two-tail ECDF calibration
- Added trimmed class calibration and hybrid rank fusion
- Added raw coordinate signal audit and label contamination diagnostics
