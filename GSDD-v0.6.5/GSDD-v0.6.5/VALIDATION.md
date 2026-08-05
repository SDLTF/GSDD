# Validation

Completed in the build environment:

- Python compilation for every `.py` file
- YAML parsing for formal and smoke configurations
- CPU synthetic end-to-end smoke run
- matched, shuffled and no-trigger factorial graph construction
- verification that generator history includes:
  - `generic_clean_cap_loss`
  - `generic_selectivity_loss`
  - matched/shuffled poisoned probabilities
  - matched/shuffled clean-control probabilities
  - matched/shuffled selectivity
  - target-similarity excess
- aggregate generation using a scalar JSON selected candidate
- selected-candidate parsing design avoids the PowerShell 5.1 top-level-array bug from v0.6.4
- shared model initialization and deterministic seed path retained

Formal Cora experiments remain CUDA-only and must be validated on the user's Windows CUDA environment.
