# Changelog

## v0.6.3

- Replaced the invalid clean-label `full - trigger-only` comparison with a model × trigger factorial audit
- Added automatic low-baseline target-class scanning
- Added victim-shuffled triggers that preserve exact trigger marginals and topology
- Added clean-label functional binding gates
- Added matched-vs-shuffled spectral interaction diagnostics
- Restricted detection candidates to same-class labeled training nodes
- Capped Cora poison counts at 12 to preserve at least eight same-class controls
- Added automatic pilot selection, multiseed expansion, aggregation, and packaging

## v0.6.2.1

- Fixed DPGBA prototype-mixture dimension mismatch
