$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand `
    -Command "python run_repro_audit.py --config configs/cora_repro_audit.yaml --seed 3407 --device cpu --name gsdd_v051_cpu_reference --strict" `
    -LogPath "results\latest_cpu_reference_seed3407.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_result_set.ps1 `
    -NamePrefix "gsdd_v051_cpu_reference" `
    -ArchiveName "gsdd_v051_cpu_reference_seed3407" -Force
