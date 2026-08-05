$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python run_gsdd_v05.py --config configs/smoke_v05.yaml" `
    -LogPath "results\latest_smoke_v05.log"

& (Join-Path $PSScriptRoot "package_latest_result.ps1") `
    -NamePrefix "gsdd_v05_smoke" `
    -Force

Write-Host "[GSDD] v0.5 paired-DID smoke test completed."
