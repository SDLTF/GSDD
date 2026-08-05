$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$seeds = @(1027, 2026, 3407)
foreach ($seed in $seeds) {
    Write-Host ("=== Seed {0} ===" -f $seed)
    Invoke-GsddCommand `
        -Command "python run_gsdd_v02.py --config configs/cora_low_poison.yaml --seed $seed --name gsdd_v02_cora_low_poison_multiseed" `
        -LogPath ("results\multiseed_{0}.log" -f $seed)
}
Invoke-GsddCommand `
    -Command "python collect_results.py --results-root results --output results\aggregate_results.csv" `
    -LogPath "results\latest_collect_results.log"
& (Join-Path $PSScriptRoot "package_result_set.ps1") -NamePrefix "gsdd_v02_cora_low_poison_multiseed" -ArchiveName "gsdd_v02_low_poison_multiseed" -Force
Write-Host "[GSDD] Multi-seed experiment completed and packaged."
