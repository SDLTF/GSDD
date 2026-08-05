# Validation

The following checks were completed in the build environment:

- Python compilation for every `.py` file
- YAML parsing for the v0.6.4 formal and smoke configurations
- CPU synthetic smoke run for generic mode
- CPU synthetic smoke run for contextual mode
- Verification that generic histories contain:
  - `generic_shuffled_target_loss`
  - `generic_consistency_loss`
- Verification that contextual histories contain:
  - `contextual_pair_loss`
  - `contextual_pair_gap`
  - `contextual_relation_loss`
  - `contextual_feature_std`
- clean/poison shared-initialization model training
- matched, shuffled, and no-trigger factorial inference
- generic and contextual spectral diagnostics
- dual-mode aggregate generation and candidate selection
- exact trigger-marginal and topology equality checks inherited from v0.6.3

Formal Cora experiments are CUDA-only and must be validated on the user's Windows CUDA environment.
