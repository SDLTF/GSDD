$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Invoke-GsddCommand `
    -Command "python run_repro_audit.py --config configs/smoke_v051.yaml --strict" `
    -LogPath "results\latest_repro_smoke.log"
Write-Host "[GSDD] v0.5.1 reproducibility smoke test completed."
