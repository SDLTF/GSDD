. "$PSScriptRoot\common.ps1"
Assert-Environment

$Dataset = "Cora"
$Attack = "SBA"
$Seed = 1027
$RunId = "Cora_SBA_seed1027"
$Repo = Join-Path $script:ProjectRoot "external\DShield-Official"
$Node = Join-Path $Repo "NodeClassificationTasks"
$Artifact = Join-Path $script:ProjectRoot "artifacts\official_attacks\$RunId"
$ResultRoot = Join-Path $script:ProjectRoot "results\$RunId"
$GsddOut = Join-Path $ResultRoot "GSDD"

if (-not (Test-Path (Join-Path $Artifact "artifact.pt"))) {
    throw "Missing existing artifact: $Artifact"
}

& $script:Python (Join-Path $script:ProjectRoot "tools\repair_artifact_v104.py") --artifact $Artifact
if ($LASTEXITCODE -ne 0) { throw "Artifact repair failed" }

Invoke-Logged (Join-Path $ResultRoot "gsdd_detection.log") $script:ProjectRoot @(
    $script:Python,
    "run_gsdd_on_artifact.py",
    "--artifact", $Artifact,
    "--output", $GsddOut,
    "--seed", $Seed,
    "--supervised-epochs", 3,
    "--ssl-epochs", 3,
    "--filter-fraction", 0.01
)

$base = @(
    $script:Python,
    "main.py",
    "--seed=$Seed",
    "--model=GCN",
    "--dataset=$Dataset",
    "--benign_epochs=3",
    "--trigger_size=3",
    "--vs_number=0",
    "--use_vs_number",
    "--target_class=1",
    "--device_id=0",
    "--selection_method=none",
    "--attack_method=$Attack",
    "--defense_method=none",
    "--gsdd_load_artifact=$Artifact",
    "--gsdd_train_idx_override=$(Join-Path $GsddOut 'filtered_train_idx.pt')"
)
$GsddLog = Join-Path $ResultRoot "gsdd_defense.log"
Invoke-Logged $GsddLog $Node $base
& $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") `
    --log $GsddLog `
    --output (Join-Path $GsddOut "official_metrics.json") `
    --dataset $Dataset `
    --attack $Attack `
    --defense GSDD `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Metric parsing failed" }

Write-Host "[GSDD-Bench] resumed Cora SBA from the existing artifact." -ForegroundColor Green
