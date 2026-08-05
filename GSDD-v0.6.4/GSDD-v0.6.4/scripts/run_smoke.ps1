$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v02.py --config configs/smoke.yaml" `
    -LogPath "results\latest_smoke.log"

& (Join-Path $PSScriptRoot "package_latest_result.ps1") -NamePrefix "gsdd_v02_smoke" -Force
Write-Host "[GSDD] v0.2 smoke test completed and packaged."
