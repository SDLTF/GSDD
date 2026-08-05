# Changelog

## 0.6.2

- added Attack Validity Repair Round 2 candidate sweep
- added target-class and selection-method CLI overrides
- added UGBA dirty-label poison-count and target-class screening
- changed DPGBA repair candidates to clean-label target-class poisoning
- added mixed target/background prototype bank for DPGBA-style generation
- added configurable `distribution_target_prototype_fraction`
- strengthened generator binding and clean-control penalties in repair configs
- added automatic pilot ranking and valid-only multiseed expansion
- added `repair2_candidate_ranking.csv` and `selected_candidates.json`
- retained the unchanged hard functional-backdoor admission gate
- retained CUDA-only formal execution
