$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "common.ps1")
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$prefix = "gsdd_v065_generic_selective"
$config = "configs/cora_generic_selective_repair.yaml"
$target = 5
$poison = 12
$seed = 1027

$candidates = @(
    @{Tag="balanced"; Trigger=3; CapW=16; Cap=0.15; SelW=10; Margin=0.40; SimW=1.5; SimAllow=0.05; Raw=0.60; Proto=0.25; Rounds=6; PoisonW=5; ShuffleW=4},
    @{Tag="clean_strong"; Trigger=3; CapW=24; Cap=0.12; SelW=14; Margin=0.45; SimW=2.0; SimAllow=0.03; Raw=0.55; Proto=0.20; Rounds=7; PoisonW=6; ShuffleW=4},
    @{Tag="subtle"; Trigger=3; CapW=20; Cap=0.12; SelW=12; Margin=0.45; SimW=3.0; SimAllow=0.00; Raw=0.48; Proto=0.10; Rounds=7; PoisonW=6; ShuffleW=5},
    @{Tag="attack_preserve"; Trigger=3; CapW=18; Cap=0.15; SelW=12; Margin=0.45; SimW=1.5; SimAllow=0.04; Raw=0.58; Proto=0.20; Rounds=8; PoisonW=7; ShuffleW=5},
    @{Tag="compact"; Trigger=2; CapW=20; Cap=0.13; SelW=12; Margin=0.42; SimW=2.0; SimAllow=0.03; Raw=0.58; Proto=0.20; Rounds=8; PoisonW=7; ShuffleW=5}
)

foreach ($c in $candidates) {
    $name = "${prefix}_$($c.Tag)_t${target}_pc${poison}_ts$($c.Trigger)"
    $command = "python run_gsdd_v065.py --config $config --target-class $target --poison-count $poison --trigger-size $($c.Trigger) --clean-cap-weight $($c.CapW) --clean-probability-cap $($c.Cap) --selectivity-weight $($c.SelW) --selectivity-margin $($c.Margin) --target-similarity-weight $($c.SimW) --target-similarity-allowance $($c.SimAllow) --raw-blend $($c.Raw) --target-prototype-fraction $($c.Proto) --outer-rounds $($c.Rounds) --poison-target-weight $($c.PoisonW) --shuffled-target-weight $($c.ShuffleW) --seed $seed --name $name --device cuda"
    Invoke-GsddCommand -Command $command -LogPath "results\latest_${name}_seed${seed}.log"
}

$aggregateDir = "results\generic_selective_repair_aggregate"
Invoke-GsddCommand `
    -Command "python aggregate_generic_selective_repair.py --prefix $prefix --output-dir $aggregateDir --pilot-seed $seed" `
    -LogPath "results\latest_v065_generic_selective_pilot_aggregate.log"

powershell.exe -ExecutionPolicy Bypass -File .\scripts\package_latest_result.ps1 `
    -ResultDir .\results\generic_selective_repair_aggregate -Force
Write-Host "[GSDD] v0.6.5 pilot completed."
