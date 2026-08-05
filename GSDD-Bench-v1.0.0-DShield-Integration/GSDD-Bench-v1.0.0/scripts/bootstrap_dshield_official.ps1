param([string]$Repository="https://github.com/csyuhao/DShield-Official.git",[string]$Ref="a1fab6a")
. "$PSScriptRoot\common.ps1"
Assert-Environment
$Repo=Join-Path $script:ProjectRoot "external\DShield-Official"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required to fetch DShield-Official." }
if (-not (Test-Path (Join-Path $Repo ".git"))) { git clone $Repository $Repo; if ($LASTEXITCODE -ne 0) { throw "git clone failed" } }
git -C $Repo fetch --all --tags; if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }
git -C $Repo checkout $Ref; if ($LASTEXITCODE -ne 0) { throw "git checkout failed: $Ref" }
$Commit=(git -C $Repo rev-parse HEAD).Trim(); $Commit | Set-Content -Encoding UTF8 (Join-Path $script:ProjectRoot "provenance\dshield_commit.lock.txt")
& $script:Python (Join-Path $script:ProjectRoot "tools\patch_dshield_py313.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield compatibility patch failed" }
& $script:Python (Join-Path $script:ProjectRoot "tools\validate_dshield_checkout.py") --repo $Repo
if ($LASTEXITCODE -ne 0) { throw "DShield checkout validation failed" }
Write-Host "[GSDD-Bench] DShield-Official ready at commit $Commit" -ForegroundColor Green
