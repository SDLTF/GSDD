$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand `
    -Command "python run_repro_audit.py --config configs/cora_repro_audit.yaml --seed 3407 --name gsdd_v051_strict_cuda --strict" `
    -LogPath "results\latest_strict_cuda_seed3407.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix "gsdd_v051_strict_cuda" `
    -ArchiveName "gsdd_v051_strict_cuda_seed3407" -Force
