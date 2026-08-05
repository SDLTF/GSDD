# GSDD-Bench v1.0.2 logging hotfix

This hotfix fixes traceback truncation under Windows PowerShell 5.1.

## Symptom

The console stops at:

```text
python.exe : Traceback (most recent call last):
NativeCommandError
```

The first Python stderr line was incorrectly promoted to a terminating
PowerShell error because the project uses `ErrorActionPreference=Stop`.
The underlying Python exception was therefore hidden.

## Apply

Copy `scripts/common.ps1` into the existing project and overwrite it.
Do not recreate the virtual environment and do not rerun setup.

Then rerun:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

The full traceback will now appear in the console and in:

```text
results\Cora_SBA_seed1027\none.log
```

The script still fails correctly when Python returns a non-zero exit code;
it simply waits for and records the complete traceback first.
