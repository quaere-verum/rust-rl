#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Viewer,
    [Parameter(Mandatory)][string]$Demonstrations,
    [Parameter(Mandatory)][string]$Notices,
    [string]$OutputDirectory = 'artifacts'
)

$ErrorActionPreference = 'Stop'
$source = Split-Path -Parent $PSScriptRoot
$licence = Join-Path $source 'LICENSE'
if (-not (Test-Path -LiteralPath $licence -PathType Leaf)) {
    throw 'The project LICENSE is required to package a release'
}
$viewerFile = Get-Item -LiteralPath $Viewer
$demonstrationRoot = (Resolve-Path -LiteralPath $Demonstrations).Path
$noticeRoot = (Resolve-Path -LiteralPath $Notices).Path
$destination = [IO.Path]::GetFullPath($OutputDirectory, (Get-Location).Path)
$name = 'rust-rl-v0.1.0-windows-x86_64'
$archive = Join-Path $destination "$name.zip"
$archiveSums = Join-Path $destination "$name.sha256"
$staged = Join-Path $destination $name
if ($viewerFile.PSIsContainer -or $viewerFile.Name -ine 'rl-viewer.exe') {
    throw 'Viewer must point to rl-viewer.exe'
}
if ((Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath $archiveSums) -or
    (Test-Path -LiteralPath $staged)) {
    throw 'Use an output directory without an existing archive or staging directory'
}
if (-not (Test-Path -LiteralPath (Join-Path $noticeRoot 'THIRD-PARTY-NOTICES') -PathType Leaf)) {
    throw 'The reviewed notices directory must contain THIRD-PARTY-NOTICES'
}

$packagedRecordings = Join-Path $staged 'demonstrations'
New-Item -ItemType Directory -Path $packagedRecordings | Out-Null
foreach ($recording in 'rocket-landing', 'alpha-execution', 'alpha-execution-cvar') {
    $run = Join-Path $demonstrationRoot $recording
    $expected = Join-Path $demonstrationRoot "$recording.sha256"
    $lines = @(Get-Content -LiteralPath $expected)
    $files = @(Get-ChildItem -LiteralPath $run -File -Recurse -Force)
    $redirections = @(Get-ChildItem -LiteralPath $run -Recurse -Force -Attributes ReparsePoint)
    if ($files.Count -ne $lines.Count -or $redirections.Count -ne 0) {
        throw "The inventory for $recording differs from its recording directory"
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in $lines) {
        if ($line -notmatch '^([0-9a-f]{64})  ((?:chunks/|pages/)?[a-zA-Z0-9][a-zA-Z0-9._-]*)$') {
            throw "Invalid checksum entry in $recording"
        }
        $hash, $relative = $Matches[1], $Matches[2]
        if (-not $seen.Add($relative)) { throw "Duplicate checksum entry in $recording" }
        $inputFile = Join-Path $run $relative
        $outputFile = Join-Path $packagedRecordings "$recording/$relative"
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputFile)) | Out-Null
        [IO.File]::Copy($inputFile, $outputFile)
        if ((Get-FileHash -LiteralPath $outputFile -Algorithm SHA256).Hash -ine $hash) {
            throw "Checksum mismatch for $recording/$relative"
        }
    }
    Copy-Item -LiteralPath $expected -Destination $packagedRecordings
    Write-Host "Verified $($lines.Count) files for $recording"
}

$exe = Join-Path $staged 'rl-viewer.exe'
Copy-Item -LiteralPath $viewerFile.FullName -Destination $exe
$validation = @(& $exe --check $packagedRecordings)
if ($LASTEXITCODE -ne 0 -or @($validation -match 'complete=true').Count -ne 3) {
    throw 'The packaged viewer did not validate three complete recordings'
}
$validation | Write-Host

Copy-Item -LiteralPath (Join-Path $source 'docs/portable-readme.txt') -Destination (Join-Path $staged 'README.txt')
Copy-Item -LiteralPath $licence -Destination $staged
foreach ($entry in Get-ChildItem -LiteralPath $noticeRoot -Force) {
    if (Test-Path -LiteralPath (Join-Path $staged $entry.Name)) {
        throw "Unexpected collision in the notices directory"
    }
    Copy-Item -LiteralPath $entry.FullName -Destination $staged -Recurse
}

Compress-Archive -LiteralPath $staged -DestinationPath $archive -CompressionLevel Optimal
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $archiveSums -Value "$digest  $name.zip"
Get-Item -LiteralPath $archive, $archiveSums | Select-Object Name, Length
