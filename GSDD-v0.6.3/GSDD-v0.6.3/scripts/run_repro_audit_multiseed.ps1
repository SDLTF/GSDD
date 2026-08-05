$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$seeds = @(1027, 2026, 3407)
foreach ($seed in $seeds) {
    $logPath = "results\latest_repro_seed${seed}.log"
    $command = "python run_repro_audit.py --config configs/cora_repro_audit.yaml --seed $seed"
    Invoke-GsddCommand -Command $command -LogPath $logPath
}

Invoke-GsddCommand `
    -Command "python aggregate_repro_audit.py --prefix gsdd_v051_cora_repro --output-dir results/repro_audit_aggregate" `
    -LogPath "results\latest_repro_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\repro_audit_aggregate -Force
powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix "gsdd_v051_cora_repro" `
    -ArchiveName "gsdd_v051_repro_multiseed" -Force

Write-Host "[GSDD] Reproducibility audit completed."
Write-Host "[GSDD] Upload artifacts\repro_audit_aggregate.zip first."
