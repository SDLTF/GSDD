Set-StrictMode -Version Latest

function Invoke-GsddCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $false)]
        [string]$LogPath
    )

    if (-not $env:ComSpec) {
        throw "ComSpec is unavailable. These Windows scripts require cmd.exe."
    }

    if ($LogPath) {
        $parent = Split-Path -Parent $LogPath
        if ($parent) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }

        # cmd.exe performs stderr/stdout merging before the stream reaches
        # Windows PowerShell 5.1. This prevents ordinary Python warnings from
        # being converted into terminating NativeCommandError records.
        & $env:ComSpec /d /s /c "$Command 2>&1" |
            Tee-Object -FilePath $LogPath
    }
    else {
        & $env:ComSpec /d /s /c "$Command 2>&1"
    }

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        if ($LogPath) {
            throw "Command failed with exit code $exitCode. See $LogPath"
        }
        throw ("Command failed with exit code {0}: {1}" -f $exitCode, $Command)
    }
}
