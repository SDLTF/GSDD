$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$counts = @(4, 8, 12, 20)
foreach ($count in $counts) {
    $name = "gsdd_v02_calibration_sweep_p$count"
    Write-Host ("=== Poison count {0} ===" -f $count)
    Invoke-GsddCommand `
        -Command "python run_gsdd_v02.py --config configs/cora_fast.yaml --poison-count $count --name $name" `
        -LogPath ("results\calibration_sweep_p{0}.log" -f $count)
}

Invoke-GsddCommand `
    -Command "python collect_results.py --results-root results --output results\aggregate_results.csv" `
    -LogPath "results\latest_collect_results.log"

& (Join-Path $PSScriptRoot "package_result_set.ps1") `
    -NamePrefix "gsdd_v02_calibration_sweep_" `
    -ArchiveName "gsdd_v02_calibration_sweep_fast" `
    -Force
Write-Host "[GSDD] Calibration sweep completed and packaged."
