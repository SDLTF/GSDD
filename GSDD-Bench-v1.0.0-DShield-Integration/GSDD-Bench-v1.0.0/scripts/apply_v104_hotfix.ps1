Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing virtual environment: $Python" }

Set-Location $ProjectRoot

$ExternalNode = Join-Path $ProjectRoot "external\DShield-Official\NodeClassificationTasks"
if (Test-Path $ExternalNode) {
    Copy-Item -Force (Join-Path $ProjectRoot "tools\gsdd_bench_export.py") (Join-Path $ExternalNode "gsdd_bench_export.py")
    Write-Host "[GSDD-Bench] updated DShield artifact loader/exporter" -ForegroundColor Cyan
}

$ArtifactRoot = Join-Path $ProjectRoot "artifacts\official_attacks"
if (Test-Path $ArtifactRoot) {
    $ArtifactDirs = @(Get-ChildItem -Path $ArtifactRoot -Directory)
    foreach ($ArtifactDir in $ArtifactDirs) {
        $ArtifactFile = Join-Path $ArtifactDir.FullName "artifact.pt"
        if (-not (Test-Path $ArtifactFile)) { continue }

        Write-Host "[GSDD-Bench] repairing artifact: $($ArtifactDir.FullName)" -ForegroundColor Cyan
        & $Python (Join-Path $ProjectRoot "tools\repair_artifact_v104.py") --artifact $ArtifactDir.FullName
        $RepairExitCode = $LASTEXITCODE
        if ($RepairExitCode -ne 0) {
            throw "Artifact repair failed with exit code $RepairExitCode`: $($ArtifactDir.FullName)"
        }
    }
}

& $Python -m py_compile `
    (Join-Path $ProjectRoot "run_gsdd_on_artifact.py") `
    (Join-Path $ProjectRoot "gsdd_core\artifact.py") `
    (Join-Path $ProjectRoot "tools\gsdd_bench_export.py") `
    (Join-Path $ProjectRoot "tools\repair_artifact_v104.py")
if ($LASTEXITCODE -ne 0) { throw "Python compile validation failed" }

Write-Host "[GSDD-Bench] v1.0.4 node-injection artifact hotfix applied." -ForegroundColor Green
