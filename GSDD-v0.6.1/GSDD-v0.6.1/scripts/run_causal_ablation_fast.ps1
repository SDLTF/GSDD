$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$seed = 1027
$modes = @("none", "label_only", "trigger_only", "full")
foreach ($mode in $modes) {
    Write-Host ("=== Fast causal ablation: seed={0}, mode={1} ===" -f $seed, $mode)
    $name = "gsdd_v04_cora_causal_ablation_fast_" + $mode
    Invoke-GsddCommand `
        -Command "python run_gsdd_v04.py --config configs/cora_causal_ablation.yaml --seed $seed --ablation-mode $mode --name $name" `
        -LogPath ("results\causal_ablation_fast_{0}.log" -f $mode)
}
Write-Host "[GSDD] Fast causal ablation finished."
