$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
$pilotSeed = 1027
$prefix = "gsdd_v064_dual_cl"
$scanPath = "results\dual_clean_label_target_scan.json"
$aggregateDir = "results\dual_clean_label_aggregate"

Invoke-GsddCommand `
    -Command "python scan_clean_label_targets.py --config configs/cora_dual_clean_label.yaml --seed $pilotSeed --device cuda --output $scanPath" `
    -LogPath "results\latest_v064_target_scan.log"
$scan = Get-Content $scanPath -Raw | ConvertFrom-Json
$targets = @($scan.selected_target_classes)
if ($targets.Count -eq 0) { throw "Target scan selected no target class." }
$target = [int]$targets[0]
Write-Host ("[GSDD] v0.6.4 target class: {0}" -f $target)

# Generic branch: transfer across victim-trigger pairings.
$genericCandidates = @(
    @{ Poison = 8;  Trigger = 3; Tag = "pc8_ts3" },
    @{ Poison = 12; Trigger = 3; Tag = "pc12_ts3" },
    @{ Poison = 12; Trigger = 4; Tag = "pc12_ts4" }
)
foreach ($candidate in $genericCandidates) {
    $poison = [int]$candidate.Poison
    $trigger = [int]$candidate.Trigger
    $tag = [string]$candidate.Tag
    $name = "${prefix}_generic_t${target}_${tag}"
    $command = "python run_gsdd_v064.py --config configs/cora_dual_clean_label.yaml --attack-mode generic --target-class $target --poison-count $poison --trigger-size $trigger --seed $pilotSeed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
}

# Contextual branch: matched trigger must beat shuffled trigger.
$contextualCandidates = @(
    @{ Poison = 8;  Trigger = 3; Pair = 4.0;  Tag = "pc8_ts3_pw4" },
    @{ Poison = 12; Trigger = 3; Pair = 6.0;  Tag = "pc12_ts3_pw6" },
    @{ Poison = 12; Trigger = 3; Pair = 10.0; Tag = "pc12_ts3_pw10" }
)
foreach ($candidate in $contextualCandidates) {
    $poison = [int]$candidate.Poison
    $trigger = [int]$candidate.Trigger
    $pair = [double]$candidate.Pair
    $tag = [string]$candidate.Tag
    $name = "${prefix}_contextual_t${target}_${tag}"
    $command = "python run_gsdd_v064.py --config configs/cora_dual_clean_label.yaml --attack-mode contextual --pair-weight $pair --target-class $target --poison-count $poison --trigger-size $trigger --seed $pilotSeed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${pilotSeed}.log"
}

Invoke-GsddCommand `
    -Command "python aggregate_dual_clean_label.py --prefix $prefix --output-dir $aggregateDir --pilot-seed $pilotSeed" `
    -LogPath "results\latest_v064_dual_pilot_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\dual_clean_label_aggregate -Force
Write-Host "[GSDD] v0.6.4 pilot completed. Upload artifacts\dual_clean_label_aggregate.zip."
