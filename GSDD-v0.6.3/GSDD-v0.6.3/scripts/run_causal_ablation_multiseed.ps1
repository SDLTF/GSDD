$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$seeds = @(1027, 2026, 3407)
$modes = @("none", "label_only", "trigger_only", "full")

foreach ($seed in $seeds) {
    foreach ($mode in $modes) {
        Write-Host ("=== Causal ablation: seed={0}, mode={1} ===" -f $seed, $mode)
        $name = "gsdd_v04_cora_causal_ablation_" + $mode
        Invoke-GsddCommand `
            -Command "python run_gsdd_v04.py --config configs/cora_causal_ablation.yaml --seed $seed --ablation-mode $mode --name $name" `
            -LogPath ("results\causal_ablation_{0}_{1}.log" -f $mode, $seed)
    }
}

Invoke-GsddCommand `
    -Command "python aggregate_causal_ablation.py --results-root results --prefix gsdd_v04_cora_causal_ablation --output-dir results\causal_ablation_aggregate" `
    -LogPath "results\latest_causal_ablation_aggregate.log"

& (Join-Path $PSScriptRoot "package_result_set.ps1") `
    -NamePrefix "gsdd_v04_cora_causal_ablation" `
    -ArchiveName "gsdd_v04_causal_ablation_multiseed" `
    -Force

$aggregateZip = Join-Path "artifacts" "gsdd_v04_causal_ablation_aggregate.zip"
if (Test-Path $aggregateZip) { Remove-Item $aggregateZip -Force }
Compress-Archive -Path "results\causal_ablation_aggregate\*" -DestinationPath $aggregateZip -Force
$hash = Get-FileHash -LiteralPath $aggregateZip -Algorithm SHA256
("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $aggregateZip -Leaf)) |
    Set-Content -LiteralPath ($aggregateZip + ".sha256") -Encoding ASCII

Write-Host "[GSDD] Causal ablation completed and packaged."
