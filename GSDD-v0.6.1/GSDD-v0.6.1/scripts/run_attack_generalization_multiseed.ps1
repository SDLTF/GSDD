$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"

$families = @(
    "fixed_rare_clique",
    "ugba_style_adaptive",
    "dpgba_style_distribution"
)
$seeds = @(1027, 2026, 3407)

foreach ($family in $families) {
    foreach ($seed in $seeds) {
        $name = "gsdd_v06_cora_attack_generalization_${family}"
        $logPath = "results\latest_v06_${family}_seed${seed}.log"
        $command = "python run_gsdd_v06.py --config configs/cora_attack_generalization.yaml --attack-family $family --seed $seed --name $name --device cuda"
        Invoke-GsddCommand -Command $command -LogPath $logPath
    }
}

Invoke-GsddCommand `
    -Command "python aggregate_attack_generalization.py --prefix gsdd_v06_cora_attack_generalization --output-dir results/attack_generalization_aggregate" `
    -LogPath "results\latest_v06_attack_generalization_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\attack_generalization_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix "gsdd_v06_cora_attack_generalization" `
    -ArchiveName "gsdd_v06_attack_generalization_multiseed" -Force

Write-Host "[GSDD] v0.6 attack-family generalization completed."
Write-Host "[GSDD] Upload artifacts\attack_generalization_aggregate.zip first."
