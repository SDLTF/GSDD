$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
$pilotSeed = 1027
$prefix = "gsdd_v063_clean_label"
$scanPath = "results\clean_label_target_scan.json"
Invoke-GsddCommand `
    -Command "python scan_clean_label_targets.py --config configs/cora_clean_label_factorial.yaml --seed $pilotSeed --device cuda --output $scanPath" `
    -LogPath "results\latest_v063_target_scan.log"
$scan = Get-Content $scanPath -Raw | ConvertFrom-Json
$targets = @($scan.selected_target_classes)
foreach ($targetValue in $targets) {
    $target = [int]$targetValue
    foreach ($poison in @(4, 8, 12)) {
        $name = "${prefix}_dpgba_t${target}_clean_pc${poison}"
        $command = "python run_gsdd_v063.py --config configs/cora_clean_label_factorial.yaml --target-class $target --poison-count $poison --seed $pilotSeed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
    }
}
Invoke-GsddCommand `
    -Command "python aggregate_clean_label_factorial.py --prefix $prefix --output-dir results/clean_label_factorial_aggregate --pilot-seed $pilotSeed" `
    -LogPath "results\latest_v063_clean_label_pilot_aggregate.log"
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\clean_label_factorial_aggregate -Force
Write-Host "[GSDD] Pilot completed. Upload artifacts\clean_label_factorial_aggregate.zip."
