. "$PSScriptRoot\common.ps1"
Assert-Environment
foreach($seed in @(1027,2026,3407)) { foreach($attack in @('SBA','UGBA','GCBA')) { & "$PSScriptRoot\run_case.ps1" -Dataset Pubmed -Attack $attack -Seed $seed } }
& $script:Python (Join-Path $script:ProjectRoot "tools\aggregate_stage1.py")
Compress-Archive -Path (Join-Path $script:ProjectRoot 'artifacts\stage1_aggregate\*') -DestinationPath (Join-Path $script:ProjectRoot 'artifacts\gsdd_bench_stage1_aggregate.zip') -Force
