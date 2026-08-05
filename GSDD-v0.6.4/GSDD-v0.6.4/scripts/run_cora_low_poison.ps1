$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v02.py --config configs/cora_low_poison.yaml" `
    -LogPath "results\latest_cora_low_poison.log"

& (Join-Path $PSScriptRoot "package_latest_result.ps1") -NamePrefix "gsdd_v02_cora_low_poison" -Force
Write-Host "[GSDD] Low-poison Cora experiment completed and packaged."
