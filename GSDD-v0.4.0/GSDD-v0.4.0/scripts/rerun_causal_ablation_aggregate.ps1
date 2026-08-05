$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python aggregate_causal_ablation.py --results-root results --prefix gsdd_v04_cora_causal_ablation --output-dir results\causal_ablation_aggregate" `
    -LogPath "results\latest_causal_ablation_aggregate.log"

$aggregateZip = Join-Path "artifacts" "gsdd_v04_causal_ablation_aggregate.zip"
if (-not (Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts" | Out-Null
}
if (Test-Path $aggregateZip) {
    Remove-Item $aggregateZip -Force
}
Compress-Archive `
    -Path "results\causal_ablation_aggregate\*" `
    -DestinationPath $aggregateZip `
    -Force

$hash = Get-FileHash -LiteralPath $aggregateZip -Algorithm SHA256
("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $aggregateZip -Leaf)) |
    Set-Content -LiteralPath ($aggregateZip + ".sha256") -Encoding ASCII

Write-Host "[GSDD] Aggregate regenerated without rerunning model training."
Write-Host ("[GSDD] Archive: {0}" -f (Resolve-Path $aggregateZip))
