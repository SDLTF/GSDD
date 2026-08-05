$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"

$seeds = @(1027, 2026, 3407)

# Fixed baseline is already known to be functional; run all seeds.
foreach ($seed in $seeds) {
    $family = "fixed_rare_clique"
    $name = "gsdd_v061_cora_attack_validity_repair_${family}"
    $command = "python run_gsdd_v061.py --config configs/cora_attack_validity_repair.yaml --attack-family $family --seed $seed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_v061_${family}_seed${seed}.log"
}

# Learned families use a pilot gate. If seed 1027 fails the hard functional
# control, the remaining seeds are skipped rather than wasting GPU time on an
# invalid direct-evasion attack.
$learnedFamilies = @(
    "ugba_style_binding_aware",
    "dpgba_style_binding_aware"
)
foreach ($family in $learnedFamilies) {
    $pilotSeed = 1027
    $name = "gsdd_v061_cora_attack_validity_repair_${family}"
    $pilotCommand = "python run_gsdd_v061.py --config configs/cora_attack_validity_repair.yaml --attack-family $family --seed $pilotSeed --name $name --device cuda"
    Invoke-GsddCommand -Command $pilotCommand -LogPath "results\latest_v061_${family}_seed${pilotSeed}.log"
    $isValid = (python check_latest_attack_validity.py --prefix $name --seed $pilotSeed).Trim().ToLowerInvariant()
    if ($isValid -eq "true") {
        Write-Host "[GSDD] $family pilot passed; expanding to seeds 2026 and 3407."
        foreach ($seed in @(2026, 3407)) {
            $command = "python run_gsdd_v061.py --config configs/cora_attack_validity_repair.yaml --attack-family $family --seed $seed --name $name --device cuda"
            Invoke-GsddCommand -Command $command -LogPath "results\latest_v061_${family}_seed${seed}.log"
        }
    }
    else {
        Write-Warning "$family pilot is an invalid functional backdoor. Remaining seeds are skipped."
    }
}

Invoke-GsddCommand `
    -Command "python aggregate_attack_validity_repair.py --prefix gsdd_v061_cora_attack_validity_repair --output-dir results/attack_validity_repair_aggregate" `
    -LogPath "results\latest_v061_attack_validity_repair_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\attack_validity_repair_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix "gsdd_v061_cora_attack_validity_repair" `
    -ArchiveName "gsdd_v061_attack_validity_repair_multiseed" -Force

Write-Host "[GSDD] v0.6.1 attack-validity repair completed."
Write-Host "[GSDD] Upload artifacts\attack_validity_repair_aggregate.zip first."
