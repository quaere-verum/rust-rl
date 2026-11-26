param(
    [Parameter(Mandatory = $true)][string]$Recording,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Recording).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
$files = @(Get-ChildItem -LiteralPath $root -File -Recurse | Sort-Object FullName)
$lines = [Collections.Generic.List[string]]::new()
$bytes = 0L
for ($i = 0; $i -lt $files.Count; $i++) {
    $file = $files[$i]
    $relative = $file.FullName.Substring($root.Length + 1).Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $lines.Add("$hash  $relative")
    $bytes += $file.Length
    if (($i + 1) % 250 -eq 0 -or $i + 1 -eq $files.Count) {
        Write-Host "Hashed $($i + 1)/$($files.Count) files"
    }
}
[IO.File]::WriteAllText((Join-Path (Get-Location) $Output), (($lines -join "`n") + "`n"), [Text.UTF8Encoding]::new($false))
[pscustomobject]@{
    Recording = $root
    Files = $files.Count
    Bytes = $bytes
    ManifestSHA256 = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
}
