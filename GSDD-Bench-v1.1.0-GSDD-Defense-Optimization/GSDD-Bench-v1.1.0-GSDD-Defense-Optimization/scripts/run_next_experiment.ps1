& "$PSScriptRoot\run_pubmed_gsdd_optimization.ps1"
if ($LASTEXITCODE -ne 0) { throw "GSDD v1.1.0 optimization experiment failed with exit code $LASTEXITCODE" }
