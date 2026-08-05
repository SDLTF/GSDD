# Changelog

## v0.5.1

- Adds a two-replica numerical reproducibility audit for the paired spectral DID experiment
- Enables PyTorch deterministic-algorithm controls without allowing `set_seed` to disable them
- Disables TF32 and configures cuBLAS workspace reproducibility in the Windows launchers
- Compares node-level Pearson/Spearman correlations, operational top-k overlap, AUROC/AUPRC drift, model behavior, and parameter differences
- Adds three launch modes: practical CUDA audit, strict CUDA seed-3407 audit, and strict CPU seed-3407 reference
- Adds three-seed aggregate reporting and automatic result packaging
- Keeps v0.5 paired-DID code and outputs for backward compatibility
