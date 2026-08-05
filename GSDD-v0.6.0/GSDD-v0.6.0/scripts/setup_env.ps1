$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Write-Host "[1/4] Checking Python..."
Invoke-GsddCommand -Command "python --version"

Write-Host "[2/4] Installing lightweight runtime dependencies..."
Invoke-GsddCommand -Command "python -m pip install -r requirements-runtime.txt"

Write-Host "[3/4] Checking Python modules..."
Invoke-GsddCommand -Command "python check_environment.py"

Write-Host "[4/4] Enforcing CUDA availability..."
Invoke-GsddCommand -Command "python check_cuda_required.py"

Write-Host "Environment check completed. CUDA is available."
