# GSDD-Bench v1.0.4.1 repair-script hotfix

This hotfix fixes `ModuleNotFoundError: No module named 'gsdd_core'` when
`tools/repair_artifact_v104.py` is launched by its file path.

Copy the `tools` and `scripts` directories into the GSDD-Bench project root,
overwrite the existing files, then run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\apply_v104_hotfix.ps1
```

The Python repair tool now inserts the project root into `sys.path` before
importing `gsdd_core`. The PowerShell wrapper also changes to the project root
and uses an explicit `foreach` loop with reliable native exit-code handling.
