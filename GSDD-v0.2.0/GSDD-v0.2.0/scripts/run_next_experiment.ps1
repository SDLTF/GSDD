# Recommended next experiment after GSDD-v0.1: low-contamination dirty-label Cora.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_cora_low_poison.ps1")
