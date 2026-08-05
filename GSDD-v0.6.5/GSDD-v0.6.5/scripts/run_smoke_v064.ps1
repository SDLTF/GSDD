$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"

foreach ($mode in @("generic", "contextual")) {
    $name = "gsdd_v064_smoke_${mode}"
    $command = "python run_gsdd_v064.py --config configs/smoke_v064.yaml --allow-cpu --attack-mode $mode --name $name"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}.log"
}
Write-Host "[GSDD] v0.6.4 dual smoke tests completed."
