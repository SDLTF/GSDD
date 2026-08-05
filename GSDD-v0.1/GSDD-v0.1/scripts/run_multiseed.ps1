$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$seeds = @(1027, 2026, 3407)
foreach ($seed in $seeds) {
    Write-Host "=== Seed $seed ==="
    Invoke-GsddCommand `
        -Command "python run_gsdd_v01.py --config configs/default.yaml --seed $seed --name gsdd_v01_cora_multiseed" `
        -LogPath "results\multiseed_$seed.log"
}

Invoke-GsddCommand `
    -Command "python collect_results.py --results-root results --output results\aggregate_results.csv" `
    -LogPath "results\latest_collect_results.log"

Write-Host "[GSDD] Multi-seed experiments and aggregation completed successfully."
