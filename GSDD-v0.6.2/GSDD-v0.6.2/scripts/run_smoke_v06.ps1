$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
Invoke-GsddCommand `
    -Command "python run_gsdd_v06.py --config configs/smoke_v06.yaml --attack-family ugba_style_adaptive --name gsdd_v06_smoke --device cuda" `
    -LogPath "results\latest_smoke_v06.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 -Force
Write-Host "[GSDD] v0.6 CUDA smoke test completed."
