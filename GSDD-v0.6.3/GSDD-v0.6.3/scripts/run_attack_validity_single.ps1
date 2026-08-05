param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("fixed_rare_clique", "ugba_style_binding_aware", "dpgba_style_binding_aware")]
    [string]$AttackFamily,
    [int]$Seed = 1027
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Invoke-GsddCommand -Command "python check_cuda_required.py" -LogPath "results\latest_cuda_check.log"
$name = "gsdd_v061_cora_attack_validity_repair_${AttackFamily}"
$command = "python run_gsdd_v061.py --config configs/cora_attack_validity_repair.yaml --attack-family $AttackFamily --seed $Seed --name $name --device cuda"
Invoke-GsddCommand -Command $command -LogPath "results\latest_v061_${AttackFamily}_seed${Seed}.log"
