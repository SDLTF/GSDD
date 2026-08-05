# Changelog

## 0.6.1

- replaced direct target-probability generator training with binding-aware poisoned-versus-control optimization
- added clean and label-only control surrogates sharing initialization
- added alternating poisoned-surrogate refitting
- removed learned-family victim-feature stamping
- changed DPGBA-style prototype bank from target-class-only to class-agnostic training prototypes
- added context-neighbor blending for learned trigger features
- changed topology search to optimize binding gap instead of raw target probability
- added hard functional-backdoor admission gate and `attack_validity.json`
- excluded invalid attacks from valid-only spectral aggregate statistics
- added learned-family pilot gate to avoid wasting GPU time after an invalid seed-1027 attack
- retained CUDA hard requirement in both code and scripts
