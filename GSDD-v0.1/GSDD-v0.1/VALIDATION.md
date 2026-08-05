# Local Validation Record

The packaged code was executed end-to-end with `configs/smoke.yaml` in the build environment.

## Integrity checks

- Python compilation: passed
- Synthetic graph generation: passed
- Controlled trigger injection: passed
- Sparse supervised GCN training: passed
- Sparse DGI training: passed
- Normalized Laplacian construction: passed
- Four-band Bernstein filtering: passed
- Hutchinson local spectral moments: passed
- CSV/JSON/model/plot generation: passed
- All diagnostic score values finite: passed

## Smoke result

- Clean test accuracy: `1.0000`
- Triggered test ASR: `1.0000`
- Pointwise model-JS AUROC: `0.8376`
- Spectral-relation AUROC: `0.5556`
- Transfer AUROC: `0.3034`
- Combined AUROC: `0.6325`

The smoke run supports code integrity only. It also demonstrates why the package reports each hypothesis separately: H3 showed a signal in this synthetic setting, while the current H4 transfer statistic did not. The negative H4 result is retained rather than hidden through label-guided tuning.

The exact files are stored in `example_smoke_result/`.
