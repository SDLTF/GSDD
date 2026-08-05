$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"

$pilotSeed = 1027
$prefix = "gsdd_v063_clean_label"
$aggregateDir = "results\clean_label_factorial_aggregate"
$scanPath = "results\clean_label_target_scan.json"

# Stage 1: train one clean model and choose target classes with low natural ASR.
Invoke-GsddCommand `
    -Command "python scan_clean_label_targets.py --config configs/cora_clean_label_factorial.yaml --seed $pilotSeed --device cuda --output $scanPath" `
    -LogPath "results\latest_v063_target_scan.log"

if (-not (Test-Path $scanPath)) {
    throw "Target scan output is missing: $scanPath"
}
$scan = Get-Content $scanPath -Raw | ConvertFrom-Json
$targets = @($scan.selected_target_classes)
if ($targets.Count -eq 0) {
    throw "Target scan selected no target classes."
}
Write-Host ("[GSDD] Selected low-baseline target classes: {0}" -f ($targets -join ", "))

# Stage 2: clean-label candidate pilots. Each target gets three poison counts.
$poisonCounts = @(4, 8, 12)
foreach ($targetValue in $targets) {
    $target = [int]$targetValue
    foreach ($poison in $poisonCounts) {
        $name = "${prefix}_dpgba_t${target}_clean_pc${poison}"
        $command = "python run_gsdd_v063.py --config configs/cora_clean_label_factorial.yaml --target-class $target --poison-count $poison --seed $pilotSeed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
    }
}

Invoke-GsddCommand `
    -Command "python aggregate_clean_label_factorial.py --prefix $prefix --output-dir $aggregateDir --pilot-seed $pilotSeed" `
    -LogPath "results\latest_v063_clean_label_pilot_aggregate.log"

$selectionPath = Join-Path $aggregateDir "selected_candidates.json"
if (-not (Test-Path $selectionPath)) {
    throw "Candidate selection output is missing: $selectionPath"
}
$selected = @(Get-Content $selectionPath -Raw | ConvertFrom-Json)
foreach ($item in $selected) {
    if (-not ([bool]$item.pilot_valid)) {
        Write-Warning ("No pilot passed the clean-label factorial gate. Best candidate: target={0}, poison={1}, full={2:N3}, control={3:N3}, gap={4:N3}, distance={5:N3}" -f `
            $item.target_class, $item.poison_count, $item.pilot_full_asr, `
            $item.pilot_control_asr_max, $item.pilot_binding_gap, $item.pilot_admission_distance)
        continue
    }
    $target = [int]$item.target_class
    $poison = [int]$item.poison_count
    $name = "${prefix}_dpgba_t${target}_clean_pc${poison}"
    Write-Host ("[GSDD] Pilot passed. Expanding target={0}, poison={1} to seeds 2026 and 3407." -f $target, $poison)
    foreach ($seed in @(2026, 3407)) {
        $command = "python run_gsdd_v063.py --config configs/cora_clean_label_factorial.yaml --target-class $target --poison-count $poison --seed $seed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
    }
}

# Stage 3: final aggregate and upload archives.
Invoke-GsddCommand `
    -Command "python aggregate_clean_label_factorial.py --prefix $prefix --output-dir $aggregateDir --pilot-seed $pilotSeed" `
    -LogPath "results\latest_v063_clean_label_final_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\clean_label_factorial_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix $prefix `
    -ArchiveName "gsdd_v063_clean_label_factorial_runs" -Force

Write-Host "[GSDD] v0.6.3 Clean-label Factorial Audit completed."
Write-Host "[GSDD] Upload artifacts\clean_label_factorial_aggregate.zip first."
