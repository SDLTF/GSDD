$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_dual_clean_label_pilot.ps1
if ($LASTEXITCODE -ne 0) { throw ("Pilot failed with exit code {0}" -f $LASTEXITCODE) }

$prefix = "gsdd_v064_dual_cl"
$aggregateDir = "results\dual_clean_label_aggregate"
$selectionPath = Join-Path $aggregateDir "selected_candidates.json"
if (-not (Test-Path $selectionPath)) { throw "Missing candidate selection: $selectionPath" }
$selected = @(Get-Content $selectionPath -Raw | ConvertFrom-Json)

foreach ($item in $selected) {
    $mode = [string]$item.attack_mode
    if (-not ([bool]$item.pilot_valid)) {
        Write-Warning ("No {0} pilot passed. Best distance={1:N3}, full={2:N3}, control={3:N3}, gap={4:N3}" -f `
            $mode, $item.pilot_admission_distance, $item.pilot_full_asr, `
            $item.pilot_control_asr_max, $item.pilot_admission_gap)
        continue
    }
    $target = [int]$item.target_class
    $poison = [int]$item.poison_count
    $trigger = [int]$item.trigger_size
    $pair = [double]$item.pair_weight
    $tag = "pc${poison}_ts${trigger}"
    if ($mode -eq "contextual") { $tag = "${tag}_pw$([int]$pair)" }
    $name = "${prefix}_${mode}_t${target}_${tag}"
    foreach ($seed in @(2026, 3407)) {
        $command = "python run_gsdd_v064.py --config configs/cora_dual_clean_label.yaml --attack-mode $mode --pair-weight $pair --target-class $target --poison-count $poison --trigger-size $trigger --seed $seed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
    }
}

Invoke-GsddCommand `
    -Command "python aggregate_dual_clean_label.py --prefix $prefix --output-dir $aggregateDir --pilot-seed 1027" `
    -LogPath "results\latest_v064_dual_final_aggregate.log"
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\dual_clean_label_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix $prefix -ArchiveName "gsdd_v064_dual_clean_label_runs" -Force
Write-Host "[GSDD] v0.6.4 Dual Clean-label Audit completed."
Write-Host "[GSDD] Upload artifacts\dual_clean_label_aggregate.zip first."
