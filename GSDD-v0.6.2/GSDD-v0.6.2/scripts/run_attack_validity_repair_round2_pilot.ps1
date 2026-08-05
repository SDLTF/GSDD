$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
$seed = 1027
$prefix = "gsdd_v062_repair2"

$candidates = @(
    @{ Tag="ugba"; Family="ugba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_ugba.yaml"; Selection="dirty_label"; Target=0; Poison=2 },
    @{ Tag="ugba"; Family="ugba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_ugba.yaml"; Selection="dirty_label"; Target=0; Poison=3 },
    @{ Tag="ugba"; Family="ugba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_ugba.yaml"; Selection="dirty_label"; Target=1; Poison=3 },
    @{ Tag="ugba"; Family="ugba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_ugba.yaml"; Selection="dirty_label"; Target=2; Poison=3 },
    @{ Tag="dpgba"; Family="dpgba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_dpgba.yaml"; Selection="clean_label"; Target=0; Poison=8 },
    @{ Tag="dpgba"; Family="dpgba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_dpgba.yaml"; Selection="clean_label"; Target=0; Poison=12 },
    @{ Tag="dpgba"; Family="dpgba_style_binding_aware"; Config="configs/cora_attack_validity_repair_round2_dpgba.yaml"; Selection="clean_label"; Target=0; Poison=16 }
)

foreach ($candidate in $candidates) {
    $name = "${prefix}_$($candidate.Tag)_t$($candidate.Target)_$($candidate.Selection)_pc$($candidate.Poison)"
    $command = "python run_gsdd_v062.py --config $($candidate.Config) --attack-family $($candidate.Family) --selection-method $($candidate.Selection) --target-class $($candidate.Target) --poison-count $($candidate.Poison) --seed $seed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
}

Invoke-GsddCommand `
    -Command "python aggregate_attack_validity_repair_round2.py --prefix $prefix --output-dir results/attack_validity_repair_round2_aggregate" `
    -LogPath "results\latest_v062_repair2_pilot_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\attack_validity_repair_round2_aggregate -Force
Write-Host "[GSDD] Pilot completed. Upload artifacts\attack_validity_repair_round2_aggregate.zip."
