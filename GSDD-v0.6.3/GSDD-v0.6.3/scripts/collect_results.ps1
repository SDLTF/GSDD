$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python collect_results.py --results-root results --output results\aggregate_results.csv" `
    -LogPath "results\latest_collect_results.log"

Write-Host "[GSDD] Result aggregation completed successfully."
