$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"

Write-Host "[1/3] Checking Python..."
Invoke-GsddCommand -Command "python --version"

Write-Host "[2/3] Checking existing PyTorch installation..."
Invoke-GsddCommand -Command 'python -c "import torch; print(''torch='', torch.__version__); print(''cuda='', torch.cuda.is_available()); print(''torch_cuda='', torch.version.cuda); print(''device='', torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''CPU'')"'

Write-Host "[3/3] Installing lightweight runtime dependencies..."
Invoke-GsddCommand -Command "python -m pip install -r requirements-runtime.txt"
Invoke-GsddCommand -Command "python check_environment.py"

Write-Host "Environment check completed."
