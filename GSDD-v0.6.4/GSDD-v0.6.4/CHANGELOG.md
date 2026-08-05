# Changelog

## v0.6.4

- Split clean-label attacks into generic and contextual modes
- Added shuffled-trigger target loss and cross-victim consistency for generic triggers
- Added matched-vs-shuffled pair margin loss for contextual triggers
- Added context-relation and trigger-diversity regularization
- Added separate functional admission gates for the two attack modes
- Added generic trigger-vs-no-trigger spectral interaction diagnostics
- Added six-candidate pilot search and per-mode multiseed expansion
- Added dual-mode aggregation and packaging

## v0.6.3

- Replaced the invalid clean-label `full - trigger-only` comparison with a model × trigger factorial audit
- Added victim-shuffled controls and low-baseline target scanning

## v0.6.2.1

- Fixed DPGBA prototype-mixture dimension mismatch
