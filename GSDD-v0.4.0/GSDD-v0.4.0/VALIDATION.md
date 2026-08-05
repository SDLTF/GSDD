# Validation

GSDD-v0.4.0 was checked in the local Linux/Python environment.

## Static validation

- All Python modules compile with `python -m py_compile`
- All four ablation modes are accepted by the CLI
- The zero-sum shape projection and covariance-score paths execute
- The aggregate script reads all four conditions and writes factorial contrasts

## Runtime smoke validation

A synthetic graph smoke run completed for:

- `none`
- `label_only`
- `trigger_only`
- `full`

Observed behavior:

- all four runs completed without exceptions
- clean accuracy was 1.0 in the synthetic smoke setting
- ASR was 1.0 only in `full`
- ASR was 0.0 in the three control conditions
- node scores, plots, summaries, and aggregate tables were produced

The synthetic smoke metrics are not evidence for the research hypothesis; they only
validate execution. The decisive test is the Cora three-seed causal ablation.
