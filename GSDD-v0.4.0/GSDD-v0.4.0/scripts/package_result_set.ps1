[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$NamePrefix,

    [Parameter(Mandatory = $false)]
    [string]$ArchiveName,

    [Parameter(Mandatory = $false)]
    [switch]$IncludeModels,

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resultDirectories = @(Get-ChildItem -Path (Join-Path $repoRoot "results") -Directory |
    Where-Object { $_.Name.StartsWith($NamePrefix) } |
    Sort-Object Name)
if ($resultDirectories.Count -eq 0) {
    throw ("No result directories match prefix: {0}" -f $NamePrefix)
}
if ([string]::IsNullOrWhiteSpace($ArchiveName)) {
    $ArchiveName = $NamePrefix + "_result_set"
}

$outputDir = Join-Path $repoRoot "artifacts"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$zipPath = Join-Path $outputDir ($ArchiveName + ".zip")
$hashPath = $zipPath + ".sha256"
if ((Test-Path $zipPath) -and (-not $Force)) {
    throw ("Archive already exists: {0}. Use -Force to overwrite it." -f $zipPath)
}
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
if (Test-Path $hashPath) { Remove-Item $hashPath -Force }

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("gsdd_set_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $staging | Out-Null
try {
    $excludedExtensions = @(".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
    foreach ($directory in $resultDirectories) {
        $destinationRoot = Join-Path $staging $directory.Name
        New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
        Get-ChildItem -LiteralPath $directory.FullName -Recurse -File | ForEach-Object {
            if ((-not $IncludeModels) -and ($excludedExtensions -contains $_.Extension.ToLowerInvariant())) {
                return
            }
            $relative = $_.FullName.Substring($directory.FullName.Length).TrimStart("\", "/")
            $destination = Join-Path $destinationRoot $relative
            New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        }
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force
    $hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
    ("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $zipPath -Leaf)) | Set-Content -LiteralPath $hashPath -Encoding ASCII
    Write-Host ("[GSDD] Packaged {0} result directories: {1}" -f $resultDirectories.Count, $zipPath)
}
finally {
    if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
}
