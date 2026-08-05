$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v01.py --config configs/default.yaml" `
    -LogPath "results\latest_cora.log"

Write-Host "[GSDD] Standard Cora experiment completed successfully."
