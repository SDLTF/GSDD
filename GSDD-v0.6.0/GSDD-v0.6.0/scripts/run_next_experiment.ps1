$ErrorActionPreference = "Stop"
powershell.exe -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_attack_generalization_multiseed.ps1")
if ($LASTEXITCODE -ne 0) {
    throw ("Next experiment failed with exit code {0}" -f $LASTEXITCODE)
}
