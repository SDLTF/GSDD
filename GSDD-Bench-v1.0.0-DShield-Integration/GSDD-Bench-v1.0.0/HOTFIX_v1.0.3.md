# GSDD-Bench v1.0.3 compatibility hotfix

This hotfix fixes eager optional-extension imports in DShield-Official under CPython 3.13 and PyTorch 2.13 CUDA 13.0.

## Fixed

1. `NodeClassificationTasks/utils.py` imported `torch_scatter` solely for one sum reduction. It is replaced with `torch_geometric.utils.scatter`, which uses modern PyTorch reductions when the optional extension is absent.
2. `models/construct.py` eagerly imported `RobustGCN`, which requires `torch_sparse` even when the selected victim model is ordinary GCN. The import is now optional.
3. Added an import smoke test before rerunning formal experiments.

## Apply

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\apply_v103_hotfix.ps1
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

Do not install an arbitrary `torch_scatter` or `torch_sparse` wheel built for a different PyTorch/CUDA combination.
