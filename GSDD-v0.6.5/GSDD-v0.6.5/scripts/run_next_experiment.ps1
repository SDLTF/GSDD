$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
& powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_generic_selective_repair.ps1
if ($LASTEXITCODE -ne 0) {
    throw ("Next experiment failed with exit code {0}" -f $LASTEXITCODE)
}
