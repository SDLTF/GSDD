# GSDD-v0.6 validation

## Static validation

- all Python modules compile successfully with `compileall`
- all new PowerShell scripts were reviewed for Windows PowerShell 5.1 interpolation hazards
- formal config explicitly sets `device: cuda`
- `run_gsdd_v06.py` rejects CPU configurations and fails when CUDA is unavailable

## Internal functional validation

The container used to build the package has no CUDA device, so the shipped CUDA-only entry point could not be executed here.

To validate implementation logic without weakening the shipped policy, the core pipeline was invoked through a temporary, non-shipped CPU harness on a synthetic graph. All three families completed:

- trigger-plan construction
- four paired graph interventions
- shared-initialization GCN training
- triggered-test-graph construction
- paired DID diagnostics
- per-run output generation
- cross-family aggregation

Internal synthetic functional outcomes:

| Family | Full ASR | Trigger-only ASR | Spectral-hybrid AUROC |
|---|---:|---:|---:|
| fixed_rare_clique | 1.00 | 0.00 | 0.679 |
| ugba_style_adaptive | 0.95 | 0.00 | 0.951 |
| dpgba_style_distribution | 0.90 | 0.00 | 0.938 |

These values are implementation checks only and are not research results.

## Remaining platform validation

The following must be established on the user's Windows RTX 5060 Laptop environment:

- CUDA preflight succeeds
- CUDA sparse backward supports trigger-generator optimization
- all three families complete on Cora
- learned-family ASR remains high under the formal 300-epoch GCN configuration
- aggregate packaging completes under Windows PowerShell 5.1
