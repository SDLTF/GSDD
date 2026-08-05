[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ResultDir,

    [Parameter(Mandatory = $false)]
    [string]$NamePrefix,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "artifacts",

    [Parameter(Mandatory = $false)]
    [switch]$IncludeModels,

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($ResultDir)) {
    $directories = Get-ChildItem -Path (Join-Path $repoRoot "results") -Directory -ErrorAction Stop
    if (-not [string]::IsNullOrWhiteSpace($NamePrefix)) {
        $directories = $directories | Where-Object { $_.Name.StartsWith($NamePrefix) }
    }
    $latest = $directories | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($null -eq $latest) {
        throw ("No matching result directory was found. Prefix: {0}" -f $NamePrefix)
    }
    $sourceDir = $latest.FullName
}
else {
    $candidate = $ResultDir
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $repoRoot $candidate
    }
    $sourceDir = (Resolve-Path $candidate -ErrorAction Stop).Path
}

if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
    throw ("Result directory does not exist: {0}" -f $sourceDir)
}

$outputPath = $OutputDir
if (-not [System.IO.Path]::IsPathRooted($outputPath)) {
    $outputPath = Join-Path $repoRoot $outputPath
}
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$resultName = Split-Path $sourceDir -Leaf
$zipPath = Join-Path $outputPath ($resultName + ".zip")
$hashPath = $zipPath + ".sha256"

if ((Test-Path -LiteralPath $zipPath) -and (-not $Force)) {
    throw ("Archive already exists: {0}. Use -Force to overwrite it." -f $zipPath)
}
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
if (Test-Path -LiteralPath $hashPath) { Remove-Item -LiteralPath $hashPath -Force }

$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("gsdd_package_" + [Guid]::NewGuid().ToString("N"))
$stagingDir = Join-Path $stagingRoot $resultName
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

try {
    $excludedExtensions = @(".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
    $included = New-Object System.Collections.Generic.List[string]
    $excluded = New-Object System.Collections.Generic.List[string]

    Get-ChildItem -LiteralPath $sourceDir -Recurse -File | ForEach-Object {
        $relativePath = $_.FullName.Substring($sourceDir.Length).TrimStart("\", "/")
        $extension = $_.Extension.ToLowerInvariant()
        if ((-not $IncludeModels) -and ($excludedExtensions -contains $extension)) {
            $excluded.Add($relativePath)
            return
        }
        $destination = Join-Path $stagingDir $relativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        $included.Add($relativePath)
    }

    $manifest = [ordered]@{
        result_name = $resultName
        source_directory = $sourceDir
        packaged_at = (Get-Date).ToString("o")
        include_models = [bool]$IncludeModels
        included_file_count = $included.Count
        excluded_file_count = $excluded.Count
        included_files = @($included)
        excluded_files = @($excluded)
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stagingDir "PACKAGE_MANIFEST.json") -Encoding UTF8

    Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force
    $hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
    ("{0}  {1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path $zipPath -Leaf)) | Set-Content -LiteralPath $hashPath -Encoding ASCII

    Write-Host ("[GSDD] Packaged result: {0}" -f $zipPath)
    Write-Host ("[GSDD] SHA-256 file:   {0}" -f $hashPath)
    Write-Host ("[GSDD] Included files: {0}" -f $included.Count)
    if (-not $IncludeModels) {
        Write-Host ("[GSDD] Model files excluded: {0}" -f $excluded.Count)
    }
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
