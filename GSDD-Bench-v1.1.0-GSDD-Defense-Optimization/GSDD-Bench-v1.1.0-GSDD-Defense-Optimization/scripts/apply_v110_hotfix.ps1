. "$PSScriptRoot\common.ps1"
Assert-Environment

$Repo = Join-Path $script:ProjectRoot "external\DShield-Official"
Assert-File (Join-Path $Repo "NodeClassificationTasks\main.py")
Assert-File (Join-Path $Repo "NodeClassificationTasks\models\GCN.py")

& $script:Python (Join-Path $script:ProjectRoot "tools\patch_dshield_v110.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "GSDD v1.1.0 DShield patch failed" }

$Node = Join-Path $Repo "NodeClassificationTasks"
Push-Location $Node
try {
    & $script:Python -c "import main; from models.GCN import GCN; import inspect; assert 'node_weights' in inspect.signature(GCN.fit).parameters; print('[GSDD-Bench] v1.1.0 weighted-training import smoke test: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "DShield weighted-training import smoke test failed" }
}
finally {
    Pop-Location
}

Write-Host "[GSDD-Bench] v1.1.0 optimization patch applied." -ForegroundColor Green
