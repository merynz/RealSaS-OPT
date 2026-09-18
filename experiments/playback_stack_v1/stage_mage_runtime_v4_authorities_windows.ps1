param(
    [Parameter(Mandatory=$true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"

$H1Name = "REALSAS_MAGE_FULL_SUBJECT_RECLOSURE_20260912"
$ProductName = "MageDemo_20260914_ProductClosure_FIT2Rebind_v1_FIX3_CONVEX_RENDER_SUPPORT"

function Test-H1Root([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    return (
        (Test-Path -LiteralPath (Join-Path $Path "IRIS_H1_V2_INPUT\20260912_FULL_SUBJECT\V0.png") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "MAGE_FIT2_PIPELINE_REFIT\ARACHNE_REFIT_V1_FIX1_TEACHER_COVERAGE_SEMANTICS\PREFLIGHT\FINAL_QUALIFIED_SKELETON_IR.json") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "MAGE_FIT2_PIPELINE_REFIT\ARACHNE_REFIT_V1_FIX1_TEACHER_COVERAGE_SEMANTICS\MAIN\ARACHNE_FIT2_QUALIFIED_SKIN_IR_FIX2.json") -PathType Leaf)
    )
}

function Test-ProductRoot([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    return (
        (Test-Path -LiteralPath (Join-Path $Path "03_P1Q_FIT2\P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "02_FIT2_P1_BIND\CURRENT_FIT2_RIGGING_SURFACE_IR.json") -PathType Leaf)
    )
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
$productRoot = $null
foreach ($base in $bases) {
    $candidate = Join-Path $base $H1Name
    if (-not $h1Root -and (Test-H1Root $candidate)) {
        $h1Root = $candidate
    }

    $candidate = Join-Path $base $ProductName
    if (-not $productRoot -and (Test-ProductRoot $candidate)) {
        $productRoot = $candidate
    }

    if (-not $productRoot) {
        $level1 = @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)
        foreach ($child1 in $level1) {
            if ($child1.Name -eq $ProductName -and (Test-ProductRoot $child1.FullName)) {
                $productRoot = $child1.FullName
                break
            }
            $level2 = @(Get-ChildItem -LiteralPath $child1.FullName -Directory -ErrorAction SilentlyContinue)
            foreach ($child2 in $level2) {
                if ($child2.Name -eq $ProductName -and (Test-ProductRoot $child2.FullName)) {
                    $productRoot = $child2.FullName
                    break
                }
            }
            if ($productRoot) { break }
        }
    }
}

if (-not $h1Root) {
    throw "MAGE_WINDOWS_DRIVE_H1_ROOT_NOT_FOUND"
}
if (-not $productRoot) {
    throw "MAGE_WINDOWS_DRIVE_PRODUCT_ROOT_NOT_FOUND"
}

$destinationFull = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destinationFull) {
    Remove-Item -LiteralPath $destinationFull -Recurse -Force
}
New-Item -ItemType Directory -Path $destinationFull | Out-Null

$inputDst = Join-Path $destinationFull "input"
$p1qDst = Join-Path $destinationFull "p1q"
$mechanicsDst = Join-Path $destinationFull "mechanics"
New-Item -ItemType Directory -Path $inputDst,$p1qDst,$mechanicsDst | Out-Null

$inputRoot = Join-Path $h1Root "IRIS_H1_V2_INPUT\20260912_FULL_SUBJECT"
for ($view=0; $view -lt 8; $view++) {
    foreach ($suffix in @(".camera.json", ".png")) {
        $name = "V{0}{1}" -f $view,$suffix
        $src = Join-Path $inputRoot $name
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
            throw "MAGE_WINDOWS_DRIVE_INPUT_MISSING:$name"
        }
        Copy-Item -LiteralPath $src -Destination (Join-Path $inputDst $name) -Force
    }
}

$skeletonSrc = Join-Path $h1Root "MAGE_FIT2_PIPELINE_REFIT\ARACHNE_REFIT_V1_FIX1_TEACHER_COVERAGE_SEMANTICS\PREFLIGHT\FINAL_QUALIFIED_SKELETON_IR.json"
$skinSrc = Join-Path $h1Root "MAGE_FIT2_PIPELINE_REFIT\ARACHNE_REFIT_V1_FIX1_TEACHER_COVERAGE_SEMANTICS\MAIN\ARACHNE_FIT2_QUALIFIED_SKIN_IR_FIX2.json"
$surfaceSrc = Join-Path $productRoot "02_FIT2_P1_BIND\CURRENT_FIT2_RIGGING_SURFACE_IR.json"
Copy-Item -LiteralPath $skeletonSrc -Destination (Join-Path $mechanicsDst "FINAL_QUALIFIED_SKELETON_IR.json") -Force
Copy-Item -LiteralPath $skinSrc -Destination (Join-Path $mechanicsDst "ARACHNE_FIT2_QUALIFIED_SKIN_IR_FIX2.json") -Force
Copy-Item -LiteralPath $surfaceSrc -Destination (Join-Path $mechanicsDst "CURRENT_FIT2_RIGGING_SURFACE_IR.json") -Force

$p1qSource = Join-Path $productRoot "03_P1Q_FIT2"
$manifestName = "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
$manifestSrc = Join-Path $p1qSource $manifestName
Copy-Item -LiteralPath $manifestSrc -Destination (Join-Path $p1qDst $manifestName) -Force
$manifest = Get-Content -LiteralPath $manifestSrc -Raw | ConvertFrom-Json

$required = New-Object System.Collections.Generic.HashSet[string]
foreach ($row in $manifest.views) {
    foreach ($key in @("mesh","skin","appearance")) {
        $name = [string]$row.files.$key
        if (-not $name) { throw "MAGE_WINDOWS_DRIVE_P1Q_MANIFEST_FILE_MISSING:$key" }
        [void]$required.Add($name)
    }
}
foreach ($name in $required) {
    $src = Join-Path $p1qSource $name
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        throw "MAGE_WINDOWS_DRIVE_P1Q_FILE_MISSING:$name"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $p1qDst $name) -Force
}

$report = [ordered]@{
    schema = "RealSaS.MageRuntimeV4WindowsDriveStage.v1"
    status = "PASS__EXACT_WINDOWS_GOOGLE_DRIVE_STAGE"
    input_file_count = 16
    p1q_payload_file_count = $required.Count
    teacher_foreground_artifacts_staged = $false
    source_owner_rasters_staged = $false
    recursive_file_discovery_used = $false
    exact_named_root_discovery_only = $true
}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destinationFull "STAGE_REPORT.json") -Encoding UTF8

Write-Output "MAGE_WINDOWS_GOOGLE_DRIVE_STAGE_PASS"
Write-Output ("STAGE_ROOT=" + $destinationFull)
