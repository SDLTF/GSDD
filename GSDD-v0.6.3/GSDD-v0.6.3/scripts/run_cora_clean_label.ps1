$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v02.py --config configs/cora_clean_label.yaml" `
    -LogPath "results\latest_cora_clean_label.log"

& (Join-Path $PSScriptRoot "package_latest_result.ps1") -NamePrefix "gsdd_v02_cora_clean_label" -Force
Write-Host "[GSDD] Clean-label Cora experiment completed and packaged."
