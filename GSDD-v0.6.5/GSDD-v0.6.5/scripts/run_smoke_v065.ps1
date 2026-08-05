$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$command = "python run_gsdd_v065.py --config configs/smoke_v065.yaml --allow-cpu"
Invoke-GsddCommand -Command $command -LogPath "results\latest_smoke_v065.log"
Write-Host "[GSDD] v0.6.5 smoke test completed."
