$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Write-Host "[GSDD] Verifying CUDA before the formal v0.5 experiment..."
Invoke-GsddCommand `
    -Command 'python -c "import torch; assert torch.cuda.is_available(), ''CUDA is unavailable in this virtual environment''; print(''torch='', torch.__version__); print(''cuda='', torch.version.cuda); print(''gpu='', torch.cuda.get_device_name(0))"' `
    -LogPath "results\latest_cuda_check.log"

$seeds = @(1027, 2026, 3407)
foreach ($seed in $seeds) {
    Write-Host ("=== Paired backdoor-specific DID: seed={0} ===" -f $seed)
    Invoke-GsddCommand `
        -Command "python run_gsdd_v05.py --config configs/cora_paired_did.yaml --seed $seed --name gsdd_v05_cora_paired_did" `
        -LogPath ("results\paired_did_{0}.log" -f $seed)
}

Invoke-GsddCommand `
    -Command "python aggregate_paired_did.py --results-root results --prefix gsdd_v05_cora_paired_did --output-dir results\paired_did_aggregate" `
    -LogPath "results\latest_paired_did_aggregate.log"

& (Join-Path $PSScriptRoot "package_result_set.ps1") `
    -NamePrefix "gsdd_v05_cora_paired_did" `
    -ArchiveName "gsdd_v05_paired_did_multiseed" `
    -Force

$aggregateZip = Join-Path "artifacts" "gsdd_v05_paired_did_aggregate.zip"
if (Test-Path $aggregateZip) { Remove-Item $aggregateZip -Force }
Compress-Archive -Path "results\paired_did_aggregate\*" -DestinationPath $aggregateZip -Force
$hash = Get-FileHash -LiteralPath $aggregateZip -Algorithm SHA256
("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $aggregateZip -Leaf)) |
    Set-Content -LiteralPath ($aggregateZip + ".sha256") -Encoding ASCII

Write-Host "[GSDD] v0.5 paired-DID multiseed experiment completed and packaged."
