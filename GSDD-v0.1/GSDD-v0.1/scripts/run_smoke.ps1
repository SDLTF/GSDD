$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v01.py --config configs/smoke.yaml" `
    -LogPath "results\latest_smoke.log"

Write-Host "[GSDD] Smoke test completed successfully."
