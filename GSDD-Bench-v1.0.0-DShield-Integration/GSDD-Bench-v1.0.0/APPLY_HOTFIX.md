# Apply GSDD-Bench v1.0.1 hotfix

1. Copy this archive's contents into the existing `GSDD-Bench-v1.0.0` project root
2. Allow the files under `requirements/`, `scripts/`, and `tools/` to overwrite existing files
3. Re-run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_py313_cuda.ps1
```

Do not delete `.venv`; the current CUDA PyTorch installation can be reused.
