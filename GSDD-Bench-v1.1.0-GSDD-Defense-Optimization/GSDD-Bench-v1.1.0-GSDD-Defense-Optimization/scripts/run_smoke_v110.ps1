. "$PSScriptRoot\common.ps1"
Assert-Environment

$Repo = Join-Path $script:ProjectRoot "external\DShield-Official"
$Node = Join-Path $Repo "NodeClassificationTasks"
Assert-File (Join-Path $Node "main.py")
& $script:Python (Join-Path $script:ProjectRoot "tools\patch_dshield_v110.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield v1.1.0 patch failed" }

$Artifact = Join-Path $script:ProjectRoot "artifacts\official_attacks\Cora_SBA_seed1027"
Assert-File (Join-Path $Artifact "artifact.pt")
& $script:Python (Join-Path $script:ProjectRoot "tools\repair_artifact_v104.py") --artifact $Artifact
if ($LASTEXITCODE -ne 0) { throw "Artifact normalization failed" }

$Root = Join-Path $script:ProjectRoot "results_optimization\Cora_SBA_seed1027_smoke"
$Detection = Join-Path $Root "Detection"
Invoke-Logged (Join-Path $Root "optimized_detection.log") $script:ProjectRoot @(
    $script:Python,
    "run_gsdd_optimized_on_artifact.py",
    "--artifact", $Artifact,
    "--output", $Detection,
    "--seed", 1027,
    "--supervised-epochs", 3,
    "--ssl-epochs", 3,
    "--patience", 3,
    "--budgets", "0.01",
    "--degree-bins", 2,
    "--min-group-size", 5
)

$Base = @(
    $script:Python, "main.py", "--seed=1027", "--model=GCN", "--dataset=Cora",
    "--benign_epochs=3", "--trigger_size=3", "--vs_number=0", "--use_vs_number",
    "--target_class=1", "--device_id=0", "--selection_method=none", "--attack_method=SBA",
    "--defense_method=none", "--gsdd_load_artifact=$Artifact"
)

$Hard = Join-Path $Detection "hard_indices\robust_max_b010_train_idx.pt"
$Soft = Join-Path $Detection "soft_weights\robust_max_b010_node_weights.pt"
Assert-File $Hard
Assert-File $Soft
Invoke-Logged (Join-Path $Root "hard.log") $Node ($Base + @("--gsdd_train_idx_override=$Hard"))
Invoke-Logged (Join-Path $Root "soft.log") $Node ($Base + @("--gsdd_train_weight_override=$Soft"))

Write-Host "[GSDD-Bench] v1.1.0 hard/soft smoke test: PASS" -ForegroundColor Green
