param(
 [ValidateSet('Cora','Pubmed')][string]$Dataset,
 [ValidateSet('SBA','UGBA','GCBA')][string]$Attack,
 [int]$Seed=1027,
 [switch]$Smoke
)
. "$PSScriptRoot\common.ps1"; Assert-Environment
$Repo=Join-Path $script:ProjectRoot "external\DShield-Official"; Assert-File (Join-Path $Repo "NodeClassificationTasks\main.py")
$Node=Join-Path $Repo "NodeClassificationTasks"
$RunId="$($Dataset)_$($Attack)_seed$Seed"; $ArtifactRoot=Join-Path $script:ProjectRoot "artifacts\official_attacks"; $Artifact=Join-Path $ArtifactRoot $RunId
$ResultRoot=Join-Path $script:ProjectRoot "results\$RunId"; New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null
$epochs=200; $vs=10; $trigger=3; if ($Dataset -eq 'Pubmed') { $vs=40 }
if ($Smoke) { $epochs=3; $vs=2 }
$base=@($script:Python,"main.py","--seed=$Seed","--model=GCN","--dataset=$Dataset","--benign_epochs=$epochs","--trigger_size=$trigger","--vs_number=$vs","--use_vs_number","--target_class=1","--device_id=0")
$attackArgs=@()
if ($Attack -eq 'SBA') { $attackArgs=@("--selection_method=none","--attack_method=SBA","--sba_attack_method=Rand_Samp","--sba_trigger_prob=0.5") }
elseif ($Attack -eq 'UGBA') { $te=200; if($Smoke){$te=2}; $attackArgs=@("--selection_method=cluster_degree","--attack_method=UGBA","--ugba_thrd=0.5","--ugba_trojan_epochs=$te","--ugba_inner_epochs=5","--ugba_target_loss_weight=5","--ugba_homo_loss_weight=50","--ugba_homo_boost_thrd=1.0") }
elseif ($Attack -eq 'GCBA') { $te=300; if($Smoke){$te=2}; $attackArgs=@("--selection_method=clean_label","--attack_method=GCBA","--gcba_num_hidden=512","--gcba_feat_budget=100","--gcba_trojan_epochs=$te","--gcba_ssl_tau=0.8","--gcba_tau=0.2","--gcba_edge_drop_ratio=0.5") }
# 1) Generate the official attack once and evaluate no defense.
$none=$base+$attackArgs+@("--defense_method=none","--gsdd_export_dir=$ArtifactRoot","--gsdd_run_id=$RunId")
$noneLog=Join-Path $ResultRoot "none.log"; Invoke-Logged $noneLog $Node $none
& $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") --log $noneLog --output (Join-Path $ResultRoot "none\official_metrics.json") --dataset $Dataset --attack $Attack --defense none --seed $Seed
# 2) Reuse exactly the same artifact for official DShield.
$pre=400; $fine=400; $cls=200; if($Dataset -eq 'Pubmed'){$cls=400}; if($Smoke){$pre=2;$fine=2;$cls=2}
$k3=0.1; $thresh=2.5; if($Attack -eq 'GCBA'){$k3=0.01; if($Dataset -eq 'Cora'){$thresh=1}}
$reuse=$base+@("--selection_method=none","--vs_number=0","--attack_method=$Attack","--defense_method=DShield","--gsdd_load_artifact=$Artifact","--dshield_pretrain_epochs=$pre","--dshield_finetune_epochs=$fine","--dshield_classify_epochs=$cls","--dshield_kappa1=5","--dshield_kappa2=5","--dshield_kappa3=$k3","--dshield_edge_drop_ratio=0.20","--dshield_feature_drop_ratio=0.20","--dshield_tau=0.9","--dshield_balance_factor=0.5","--dshield_classify_rounds=1","--dshield_thresh=$thresh")
$dshieldLog=Join-Path $ResultRoot "dshield.log"; Invoke-Logged $dshieldLog $Node $reuse
& $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") --log $dshieldLog --output (Join-Path $ResultRoot "DShield\official_metrics.json") --dataset $Dataset --attack $Attack --defense DShield --seed $Seed
# Normalize node-sized tensors for node-injection attacks (SBA, UGBA variants, etc.).
& $script:Python (Join-Path $script:ProjectRoot "tools\repair_artifact_v104.py") --artifact $Artifact
if ($LASTEXITCODE -ne 0) { throw "Artifact normalization failed: $Artifact" }
# 3) Run GSDD detection on the same poisoned graph.
$gsddOut=Join-Path $ResultRoot "GSDD"; $se=200; $de=200; if($Smoke){$se=3;$de=3}
Invoke-Logged (Join-Path $ResultRoot "gsdd_detection.log") $script:ProjectRoot @($script:Python,"run_gsdd_on_artifact.py","--artifact",$Artifact,"--output",$gsddOut,"--seed",$Seed,"--supervised-epochs",$se,"--ssl-epochs",$de,"--filter-fraction",0.01)
# 4) Feed GSDD's filtered training set back to official evaluation for ASR/CA.
$gsddEval=$base+@("--selection_method=none","--vs_number=0","--attack_method=$Attack","--defense_method=none","--gsdd_load_artifact=$Artifact","--gsdd_train_idx_override=$(Join-Path $gsddOut 'filtered_train_idx.pt')")
$gsddLog=Join-Path $ResultRoot "gsdd_defense.log"; Invoke-Logged $gsddLog $Node $gsddEval
& $script:Python (Join-Path $script:ProjectRoot "tools\parse_official_log.py") --log $gsddLog --output (Join-Path $gsddOut "official_metrics.json") --dataset $Dataset --attack $Attack --defense GSDD --seed $Seed
Write-Host "[GSDD-Bench] completed $RunId" -ForegroundColor Green
