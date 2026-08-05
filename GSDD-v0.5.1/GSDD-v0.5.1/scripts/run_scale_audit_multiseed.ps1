$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

$seeds = @(1027, 2026, 3407)
foreach ($seed in $seeds) {
    Write-Host ("=== Scale audit seed {0} ===" -f $seed)
    Invoke-GsddCommand `
        -Command "python run_gsdd_v03.py --config configs/cora_scale_audit.yaml --seed $seed --name gsdd_v03_cora_scale_audit_multiseed" `
        -LogPath ("results\scale_audit_{0}.log" -f $seed)
}

Invoke-GsddCommand `
    -Command "python aggregate_scale_audit.py --results-root results --prefix gsdd_v03_cora_scale_audit_multiseed --output-dir results\scale_audit_aggregate" `
    -LogPath "results\latest_scale_audit_aggregate.log"

& (Join-Path $PSScriptRoot "package_result_set.ps1") `
    -NamePrefix "gsdd_v03_cora_scale_audit_multiseed" `
    -ArchiveName "gsdd_v03_scale_audit_multiseed" `
    -Force

# Add aggregate tables to a separate small archive for convenient inspection.
$aggregateZip = Join-Path "artifacts" "gsdd_v03_scale_audit_aggregate.zip"
if (Test-Path $aggregateZip) { Remove-Item $aggregateZip -Force }
Compress-Archive -Path "results\scale_audit_aggregate\*" -DestinationPath $aggregateZip -Force
Write-Host "[GSDD] Multi-seed scale audit completed and packaged."
