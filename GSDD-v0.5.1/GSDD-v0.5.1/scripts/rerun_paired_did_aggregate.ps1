$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Invoke-GsddCommand `
    -Command "python aggregate_paired_did.py --results-root results --prefix gsdd_v05_cora_paired_did --output-dir results\paired_did_aggregate" `
    -LogPath "results\latest_paired_did_aggregate.log"

$aggregateZip = Join-Path "artifacts" "gsdd_v05_paired_did_aggregate.zip"
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null
if (Test-Path $aggregateZip) { Remove-Item $aggregateZip -Force }
Compress-Archive -Path "results\paired_did_aggregate\*" -DestinationPath $aggregateZip -Force
$hash = Get-FileHash -LiteralPath $aggregateZip -Algorithm SHA256
("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $aggregateZip -Leaf)) |
    Set-Content -LiteralPath ($aggregateZip + ".sha256") -Encoding ASCII
Write-Host "[GSDD] Paired-DID aggregate regenerated without retraining."
