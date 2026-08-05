$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v03.py --config configs/cora_scale_audit.yaml" `
    -LogPath "results\latest_scale_audit.log"

& (Join-Path $PSScriptRoot "package_latest_result.ps1") `
    -NamePrefix "gsdd_v03_cora_scale_audit" `
    -Force
Write-Host "[GSDD] Single-seed scale audit completed and packaged."
