# GSDD-Bench v1.0.1 hotfix

## Root cause

`requirements/requirements-py313-core.txt` contained the misspelled package name `umexpr`. The intended package is `numexpr`.

The v1.0.0 setup script also did not check native process exit codes after each pip command under Windows PowerShell 5.1, so it continued after dependency installation failed. Finally, it attempted to write the lock file before ensuring that `provenance/` existed.

## Recovery

Copy the hotfix over the v1.0.0 project root and run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_py313_cuda.ps1
```

The existing `.venv` is reusable. Use `-RecreateVenv` only if a clean rebuild is desired.
