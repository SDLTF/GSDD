$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v01.py --config configs/cora_fast.yaml" `
    -LogPath "results\latest_cora_fast.log"

Write-Host "[GSDD] Fast Cora experiment completed successfully."
