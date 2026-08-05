$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
Invoke-GsddCommand -Command "python run_gsdd_v062.py --config configs/smoke_v062.yaml --device cuda" -LogPath "results\latest_smoke_v062.log"
