param(
    [Parameter(Mandatory=$true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$H1Name = "REALSAS_MAGE_FULL_SUBJECT_RECLOSURE_20260912"
$RunRel = "IRIS_H1_V2_CONTINUATION_RUNS\20260912T074348Z\ZERO_SURFACE_RAW.npz"

function Test-H1RawRoot([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    return (Test-Path -LiteralPath (Join-Path $Path $RunRel) -PathType Leaf)
}

$bases = New-Object System.Collections.Generic.List[string]
foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
    $root = [string]$drive.Root
    if (-not $root) { continue }
    foreach ($candidate in @(
        (Join-Path $root "My Drive"),
        (Join-Path $root "Google Drive\My Drive"),
        (Join-Path $root "Google Drive")
    )) {
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and -not $bases.Contains($candidate)) {
            $bases.Add($candidate)
        }
    }
    if (Test-Path -LiteralPath (Join-Path $root $H1Name) -PathType Container) {
        if (-not $bases.Contains($root)) { $bases.Add($root) }
    }
}

$h1Root = $null
foreach ($base in $bases) {
    $candidate = Join-Path $base $H1Name
    if (Test-H1RawRoot $candidate) {
        $h1Root = $candidate
        break
    }
}
if (-not $h1Root) {
    throw "MAGE_WINDOWS_DRIVE_H1_RAW_ROOT_NOT_FOUND"
}

$src = Join-Path $h1Root $RunRel
$dst = [System.IO.Path]::GetFullPath($Destination)
$parent = Split-Path -Parent $dst
if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}
Copy-Item -LiteralPath $src -Destination $dst -Force
if (-not (Test-Path -LiteralPath $dst -PathType Leaf)) {
    throw "MAGE_WINDOWS_DRIVE_H1_RAW_STAGE_FAILED"
}
Write-Output "MAGE_WINDOWS_H1_RAW_ZERO_SURFACE_STAGE_PASS"
Write-Output ("RAW_ZERO_SURFACE=" + $dst)
