param(
    [int]$Seed = 1027,
    [switch]$Force
)

. "$PSScriptRoot\common.ps1"
Assert-Environment

$Repo = Join-Path $script:ProjectRoot "external\DShield-Official"
$Node = Join-Path $Repo "NodeClassificationTasks"
Assert-File (Join-Path $Node "main.py")

& $script:Python (Join-Path $script:ProjectRoot "tools\patch_dshield_v110.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield v1.1.0 patch failed" }

$Methods = @("robust_max", "fisher", "cauchy")
$Budgets = @(
    @{ Fraction = 0.005; Code = "b005" },
    @{ Fraction = 0.010; Code = "b010" },
    @{ Fraction = 0.020; Code = "b020" },
    @{ Fraction = 0.050; Code = "b050" }
)
$Attacks = @("SBA", "UGBA", "GCBA")
$ArtifactRoot = Join-Path $script:ProjectRoot "artifacts\official_attacks"
$OptimizationRoot = Join-Path $script:ProjectRoot "results_optimization"
New-Item -ItemType Directory -Force -Path $OptimizationRoot | Out-Null

foreach ($Attack in $Attacks) {
    $RunId = "Pubmed_$($Attack)_seed$Seed"
    $Artifact = Join-Path $ArtifactRoot $RunId
    Assert-File (Join-Path $Artifact "artifact.pt")

    # Normalize old node-injection artifacts in place. This is idempotent.
    & $script:Python (Join-Path $script:ProjectRoot "tools\repair_artifact_v104.py") --artifact $Artifact
    if ($LASTEXITCODE -ne 0) { throw "Artifact normalization failed: $Artifact" }

    $CaseRoot = Join-Path $OptimizationRoot $RunId
    $Detection = Join-Path $CaseRoot "Detection"
    New-Item -ItemType Directory -Force -Path $Detection | Out-Null
    $DetectionSummary = Join-Path $Detection "optimization_summary.json"

    if ($Force -or -not (Test-Path $DetectionSummary)) {
        Invoke-Logged (Join-Path $CaseRoot "optimized_detection.log") $script:ProjectRoot @(
            $script:Python,
            "run_gsdd_optimized_on_artifact.py",
            "--artifact", $Artifact,
            "--output", $Detection,
            "--seed", $Seed,
            "--supervised-epochs", 200,
            "--ssl-epochs", 200,
            "--budgets", "0.005,0.01,0.02,0.05",
            "--degree-bins", 4,
            "--min-group-size", 20,
            "--soft-strength", 6.0
        )
    }

    Assert-File $DetectionSummary

    $Base = @(
        $script:Python,
        "main.py",
        "--seed=$Seed",
        "--model=GCN",
        "--dataset=Pubmed",
        "--benign_epochs=200",
        "--trigger_size=3",
        "--vs_number=0",
        "--use_vs_number",
        "--target_class=1",
        "--device_id=0",
        "--selection_method=none",
        "--attack_method=$Attack",
        "--defense_method=none",
        "--gsdd_load_artifact=$Artifact"
    )

    foreach ($Method in $Methods) {
        foreach ($Budget in $Budgets) {
            $Code = $Budget.Code

            $HardName = "GSDD2_Hard_$($Method)_$Code"
            $HardDir = Join-Path $CaseRoot $HardName
            $HardMetrics = Join-Path $HardDir "official_metrics.json"
            $HardOverride = Join-Path $Detection "hard_indices\$($Method)_$($Code)_train_idx.pt"
            Assert-File $HardOverride
            if ($Force -or -not (Test-Path $HardMetrics)) {
                $HardLog = Join-Path $HardDir "official_eval.log"
                Invoke-Logged $HardLog $Node ($Base + @("--gsdd_train_idx_override=$HardOverride"))
                & $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") `
                    --log $HardLog --output $HardMetrics --dataset Pubmed --attack $Attack `
                    --defense $HardName --seed $Seed
                if ($LASTEXITCODE -ne 0) { throw "Metric parsing failed: $HardLog" }
            }

            $SoftName = "GSDD2_Soft_$($Method)_$Code"
            $SoftDir = Join-Path $CaseRoot $SoftName
            $SoftMetrics = Join-Path $SoftDir "official_metrics.json"
            $SoftOverride = Join-Path $Detection "soft_weights\$($Method)_$($Code)_node_weights.pt"
            Assert-File $SoftOverride
            if ($Force -or -not (Test-Path $SoftMetrics)) {
                $SoftLog = Join-Path $SoftDir "official_eval.log"
                Invoke-Logged $SoftLog $Node ($Base + @("--gsdd_train_weight_override=$SoftOverride"))
                & $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") `
                    --log $SoftLog --output $SoftMetrics --dataset Pubmed --attack $Attack `
                    --defense $SoftName --seed $Seed
                if ($LASTEXITCODE -ne 0) { throw "Metric parsing failed: $SoftLog" }
            }
        }
    }
}

& $script:Python (Join-Path $script:ProjectRoot "tools\aggregate_v110_optimization.py") --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "v1.1.0 optimization aggregation failed" }

Write-Host "[GSDD-Bench] v1.1.0 PubMed optimization completed." -ForegroundColor Green
Write-Host "Upload artifacts\gsdd_v110_optimization_aggregate.zip" -ForegroundColor Cyan
