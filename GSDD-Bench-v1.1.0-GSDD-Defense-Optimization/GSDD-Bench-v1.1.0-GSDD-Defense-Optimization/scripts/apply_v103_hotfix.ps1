$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"
Assert-Environment
$Repo = Join-Path $script:ProjectRoot "external\DShield-Official"
if (-not (Test-Path (Join-Path $Repo "NodeClassificationTasks\main.py"))) {
    throw "DShield checkout not found. Run scripts\bootstrap_dshield_official.ps1 first."
}
& $script:Python (Join-Path $script:ProjectRoot "tools\patch_dshield_py313.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield v1.0.3 compatibility patch failed" }
& $script:Python (Join-Path $script:ProjectRoot "tools\validate_dshield_checkout.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield v1.0.3 checkout validation failed" }
$NodeTasks = Join-Path $Repo "NodeClassificationTasks"
Push-Location $NodeTasks
try {
    & $script:Python -c "import main; print('[GSDD-Bench] DShield import smoke test: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "DShield import smoke test failed" }
}
finally {
    Pop-Location
}
Write-Host "[GSDD-Bench] v1.0.3 compatibility hotfix applied successfully." -ForegroundColor Green
