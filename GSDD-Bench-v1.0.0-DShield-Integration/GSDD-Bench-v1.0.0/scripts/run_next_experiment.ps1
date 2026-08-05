$ErrorActionPreference='Stop'
& "$PSScriptRoot\run_cora_smoke.ps1"
if ($LASTEXITCODE -ne 0) { throw 'Cora smoke failed' }
& "$PSScriptRoot\run_pubmed_pilot.ps1"
if ($LASTEXITCODE -ne 0) { throw 'PubMed pilot failed' }
