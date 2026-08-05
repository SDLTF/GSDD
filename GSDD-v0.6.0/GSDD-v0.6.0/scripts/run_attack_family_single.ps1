param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("fixed_rare_clique", "ugba_style_adaptive", "dpgba_style_distribution")]
    [string]$AttackFamily,

    [Parameter(Mandatory = $false)]
    [int]$Seed = 1027
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
$name = "gsdd_v06_cora_attack_generalization_${AttackFamily}"
$logPath = "results\latest_v06_${AttackFamily}_seed${Seed}.log"
$command = "python run_gsdd_v06.py --config configs/cora_attack_generalization.yaml --attack-family $AttackFamily --seed $Seed --name $name --device cuda"
Invoke-GsddCommand -Command $command -LogPath $logPath
