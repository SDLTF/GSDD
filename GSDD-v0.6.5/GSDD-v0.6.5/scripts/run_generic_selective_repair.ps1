$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_generic_selective_repair_pilot.ps1
if ($LASTEXITCODE -ne 0) { throw ("Pilot failed with exit code {0}" -f $LASTEXITCODE) }

$prefix = "gsdd_v065_generic_selective"
$aggregateDir = "results\generic_selective_repair_aggregate"
$selectionPath = Join-Path $aggregateDir "selected_candidate.json"
if (-not (Test-Path $selectionPath)) { throw "Missing candidate selection: $selectionPath" }
$item = ConvertFrom-Json -InputObject (Get-Content $selectionPath -Raw)

if ([bool]$item.pilot_valid) {
    $target = [int]$item.target_class
    $poison = [int]$item.poison_count
    $trigger = [int]$item.trigger_size
    $name = "${prefix}_selected_t${target}_pc${poison}_ts${trigger}"
    foreach ($seed in @(2026, 3407)) {
        $command = "python run_gsdd_v065.py --config configs/cora_generic_selective_repair.yaml --target-class $target --poison-count $poison --trigger-size $trigger --clean-cap-weight $($item.clean_cap_weight) --clean-probability-cap $($item.clean_probability_cap) --selectivity-weight $($item.selectivity_weight) --selectivity-margin $($item.selectivity_margin) --target-similarity-weight $($item.target_similarity_weight) --target-similarity-allowance $($item.target_similarity_allowance) --raw-blend $($item.raw_blend) --target-prototype-fraction $($item.target_prototype_fraction) --outer-rounds $($item.outer_rounds) --poison-target-weight $($item.poison_target_weight) --shuffled-target-weight $($item.shuffled_target_weight) --seed $seed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
    }
} else {
    Write-Warning ("No pilot passed. Best distance={0:N3}, full={1:N3}, control={2:N3}, DiD={3:N3}" -f `
        $item.pilot_admission_distance, $item.pilot_full_asr, `
        $item.pilot_control_asr_max, $item.pilot_generic_did)
}

Invoke-GsddCommand `
    -Command "python aggregate_generic_selective_repair.py --prefix $prefix --output-dir $aggregateDir --pilot-seed 1027" `
    -LogPath "results\latest_v065_generic_selective_final_aggregate.log"
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\generic_selective_repair_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix $prefix -ArchiveName "gsdd_v065_generic_selective_runs" -Force
Write-Host "[GSDD] v0.6.5 Generic Selective-activation Repair completed."
Write-Host "[GSDD] Upload artifacts\generic_selective_repair_aggregate.zip first."
