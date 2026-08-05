$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"

Invoke-GsddCommand `
    -Command "python run_gsdd_v063.py --config configs/smoke_v063.yaml --device cpu --allow-cpu" `
    -LogPath "results\latest_gsdd_v063_smoke.log"

Write-Host "[GSDD] v0.6.3 smoke test completed."
