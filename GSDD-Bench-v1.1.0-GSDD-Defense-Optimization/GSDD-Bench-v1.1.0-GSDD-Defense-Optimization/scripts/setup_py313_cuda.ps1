param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$RecreateVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$Step
    )
    Write-Host "[GSDD-Bench] $Step" -ForegroundColor Cyan
    & $Exe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Step failed with exit code $code"
    }
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    throw "Python launcher 'py' was not found. Install standard CPython 3.13 for Windows."
}

Invoke-NativeChecked `
    -Exe "py" `
    -Arguments @("-3.13", "-c", "import sys,sysconfig; assert sys.version_info[:2]==(3,13); assert not (sysconfig.get_config_var('Py_GIL_DISABLED') or 0)") `
    -Step "Checking standard CPython 3.13"

if ($RecreateVenv -and (Test-Path ".venv")) {
    Write-Host "[GSDD-Bench] Removing existing .venv" -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-NativeChecked -Exe "py" -Arguments @("-3.13", "-m", "venv", ".venv") -Step "Creating virtual environment"
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Requirements = Join-Path $Root "requirements\requirements-py313-core.txt"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "provenance") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "artifacts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "results") | Out-Null

Invoke-NativeChecked -Exe $Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -Step "Updating packaging tools"
Invoke-NativeChecked -Exe $Python -Arguments @("-m", "pip", "install", "torch", "--index-url", $TorchIndexUrl) -Step "Installing CUDA PyTorch"
Invoke-NativeChecked -Exe $Python -Arguments @("-m", "pip", "install", "--prefer-binary", "-r", $Requirements) -Step "Installing Python 3.13 dependencies"
Invoke-NativeChecked -Exe $Python -Arguments @("tools\check_python313.py") -Step "Validating Python interpreter"
Invoke-NativeChecked -Exe $Python -Arguments @("tools\check_cuda_required.py") -Step "Validating CUDA"
Invoke-NativeChecked -Exe $Python -Arguments @("tools\validate_environment.py") -Step "Validating imports"
Invoke-NativeChecked -Exe $Python -Arguments @("-m", "pip", "check") -Step "Checking dependency consistency"

$lockPath = Join-Path $Root "provenance\python313-installed.lock.txt"
$freezeOutput = & $Python -m pip freeze
$freezeCode = $LASTEXITCODE
if ($freezeCode -ne 0) {
    throw "Writing dependency lock failed because pip freeze exited with code $freezeCode"
}
$freezeOutput | Set-Content -Encoding UTF8 $lockPath

Write-Host "[GSDD-Bench] Python 3.13 CUDA environment ready." -ForegroundColor Green
Write-Host "[GSDD-Bench] Lock file: $lockPath" -ForegroundColor Green
