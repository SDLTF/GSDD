# Validation

The following checks were completed in the build environment:

- Python compilation for every `.py` file
- YAML parsing for every configuration file
- CPU synthetic smoke run through:
  - clean-label DPGBA-style plan generation
  - matched and victim-shuffled trigger graph construction
  - shared-initialization clean/poison model training
  - six factorial inference cells
  - clean-label validity calculation
  - spectral interaction diagnostics
  - node-level CSV and summary output
- CPU target-class scan
- aggregate generation and candidate selection
- exact matched/shuffled trigger marginal equality checks
- matched/shuffled graph topology and label equality checks

Formal Cora runs are CUDA-only and must be validated on the user's CUDA-enabled Windows environment.
