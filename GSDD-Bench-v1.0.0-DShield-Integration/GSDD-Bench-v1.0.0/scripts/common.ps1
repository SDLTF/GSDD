Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot
$script:Python = Join-Path $script:ProjectRoot ".venv\Scripts\python.exe"

function Assert-File([string]$Path) {
    if (-not (Test-Path $Path)) { throw "Missing file: $Path" }
}

function Assert-Environment {
    Assert-File $script:Python
    & $script:Python (Join-Path $script:ProjectRoot "tools\check_python313.py")
    if ($LASTEXITCODE -ne 0) { throw "Python 3.13 check failed" }
    & $script:Python (Join-Path $script:ProjectRoot "tools\check_cuda_required.py")
    if ($LASTEXITCODE -ne 0) { throw "CUDA check failed" }
}

function Invoke-Logged([string]$LogPath, [string]$WorkingDirectory, [string[]]$Command) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) { $args = $Command[1..($Command.Count - 1)] }

    "[COMMAND] $exe $($args -join ' ')" | Tee-Object -FilePath $LogPath

    Push-Location $WorkingDirectory
    $code = 1
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 converts every native stderr line into an
        # ErrorRecord. With ErrorActionPreference=Stop, a Python traceback is
        # terminated after its first line and the real exception is hidden.
        # Continue only around the native process, stringify every stream item,
        # and decide success exclusively from the native exit code.
        $ErrorActionPreference = "Continue"
        & $exe @args 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $line | Tee-Object -FilePath $LogPath -Append
        }
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }

    if ($null -eq $code) {
        throw "Native command did not expose an exit code. See $LogPath"
    }
    if ($code -ne 0) {
        throw "Command failed with exit code $code. Full stdout/stderr is in $LogPath"
    }
}
