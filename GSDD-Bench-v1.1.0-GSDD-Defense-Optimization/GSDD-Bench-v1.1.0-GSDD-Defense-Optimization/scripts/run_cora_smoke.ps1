. "$PSScriptRoot\common.ps1"
Assert-Environment
foreach($attack in @('SBA','UGBA','GCBA')) { & "$PSScriptRoot\run_case.ps1" -Dataset Cora -Attack $attack -Seed 1027 -Smoke }
& $script:Python (Join-Path $script:ProjectRoot "tools\aggregate_stage1.py")
