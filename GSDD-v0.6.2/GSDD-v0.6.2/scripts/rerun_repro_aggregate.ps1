$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python aggregate_repro_audit.py --prefix gsdd_v051_cora_repro --output-dir results/repro_audit_aggregate" `
    -LogPath "results\latest_repro_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\repro_audit_aggregate -Force
Write-Host "[GSDD] Aggregate regenerated without rerunning model training."
