$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"

$pilotSeed = 1027
$prefix = "gsdd_v062_repair2"
$aggregateDir = "results\attack_validity_repair_round2_aggregate"

# UGBA-style candidates: lower poison count and alternative target classes are
# tested to reduce label-only target-prior leakage without weakening the hard
# functional-backdoor admission rule.
$ugbaCandidates = @(
    @{ Target = 0; Poison = 2 },
    @{ Target = 0; Poison = 3 },
    @{ Target = 1; Poison = 3 },
    @{ Target = 2; Poison = 3 }
)

foreach ($candidate in $ugbaCandidates) {
    $target = [int]$candidate.Target
    $poison = [int]$candidate.Poison
    $name = "${prefix}_ugba_t${target}_dirty_pc${poison}"
    $command = "python run_gsdd_v062.py --config configs/cora_attack_validity_repair_round2_ugba.yaml --attack-family ugba_style_binding_aware --selection-method dirty_label --target-class $target --poison-count $poison --seed $pilotSeed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
}

# DPGBA-style candidates: clean-label target-class victims remove the label-only
# confounder. Increasing the number of target-class trigger examples tests
# whether the mixed prototype generator can establish functional binding.
$dpgbaCandidates = @(
    @{ Target = 0; Poison = 8 },
    @{ Target = 0; Poison = 12 },
    @{ Target = 0; Poison = 16 }
)

foreach ($candidate in $dpgbaCandidates) {
    $target = [int]$candidate.Target
    $poison = [int]$candidate.Poison
    $name = "${prefix}_dpgba_t${target}_clean_pc${poison}"
    $command = "python run_gsdd_v062.py --config configs/cora_attack_validity_repair_round2_dpgba.yaml --attack-family dpgba_style_binding_aware --selection-method clean_label --target-class $target --poison-count $poison --seed $pilotSeed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
}

Invoke-GsddCommand `
    -Command "python aggregate_attack_validity_repair_round2.py --prefix $prefix --output-dir results/attack_validity_repair_round2_aggregate" `
    -LogPath "results\latest_v062_repair2_pilot_aggregate.log"

$selectionPath = Join-Path $aggregateDir "selected_candidates.json"
if (-not (Test-Path $selectionPath)) {
    throw "Candidate selection output is missing: $selectionPath"
}
$selected = Get-Content $selectionPath -Raw | ConvertFrom-Json

foreach ($item in $selected) {
    if (-not ([bool]$item.pilot_valid)) {
        Write-Warning ("No valid pilot for {0}; multiseed expansion skipped. Best pilot: full={1:N3}, control={2:N3}, gap={3:N3}, distance={4:N3}" -f `
            $item.attack_family, $item.pilot_full_asr, $item.pilot_control_asr_max, `
            $item.pilot_binding_gap, $item.pilot_admission_distance)
        continue
    }

    $family = [string]$item.attack_family
    $target = [int]$item.target_class
    $selection = [string]$item.selection_method
    $poison = [int]$item.poison_count

    if ($family -eq "ugba_style_binding_aware") {
        $config = "configs/cora_attack_validity_repair_round2_ugba.yaml"
        $tag = "ugba"
    }
    elseif ($family -eq "dpgba_style_binding_aware") {
        $config = "configs/cora_attack_validity_repair_round2_dpgba.yaml"
        $tag = "dpgba"
    }
    else {
        Write-Warning "Unknown selected family $family; skipping."
        continue
    }

    $name = "${prefix}_${tag}_t${target}_${selection}_pc${poison}"
    Write-Host "[GSDD] Pilot passed for $family. Expanding $name to seeds 2026 and 3407."
    foreach ($seed in @(2026, 3407)) {
        $command = "python run_gsdd_v062.py --config $config --attack-family $family --selection-method $selection --target-class $target --poison-count $poison --seed $seed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
    }
}

# Re-aggregate after any multiseed expansion.
Invoke-GsddCommand `
    -Command "python aggregate_attack_validity_repair_round2.py --prefix $prefix --output-dir results/attack_validity_repair_round2_aggregate" `
    -LogPath "results\latest_v062_repair2_final_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\attack_validity_repair_round2_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix $prefix `
    -ArchiveName "gsdd_v062_attack_validity_repair_round2_runs" -Force

Write-Host "[GSDD] v0.6.2 Attack Validity Repair Round 2 completed."
Write-Host "[GSDD] Upload artifacts\attack_validity_repair_round2_aggregate.zip first."
