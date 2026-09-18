param(
    [Parameter(Mandatory=$true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$CorpusRootName = "RealSaS_MASTER_CORPUS_1024_V3"
$AssetRel = "sources\gold\kaykit_adventurers\addons\kaykit_character_pack_adventures\Characters\fbx"

$bases = New-Object System.Collections.Generic.List[string]
foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
    $root = [string]$drive.Root
    if (-not $root) { continue }
    foreach ($candidate in @(
        (Join-Path $root "My Drive"),
        (Join-Path $root "Google Drive\My Drive"),
        (Join-Path $root "Google Drive"),
        $root
    )) {
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and -not $bases.Contains($candidate)) {
            $bases.Add($candidate)
        }
    }
}

$corpusRoot = $null
foreach ($base in $bases) {
    $candidate = Join-Path $base $CorpusRootName
    $fbx = Join-Path $candidate (Join-Path $AssetRel "Mage.fbx")
    if (Test-Path -LiteralPath $fbx -PathType Leaf) {
        $corpusRoot = $candidate
        break
    }
}
if (-not $corpusRoot) {
    # Exact bounded two-level search for the canonical named corpus root only.
    foreach ($base in $bases) {
        foreach ($child1 in @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
            if ($child1.Name -eq $CorpusRootName) {
                $fbx = Join-Path $child1.FullName (Join-Path $AssetRel "Mage.fbx")
                if (Test-Path -LiteralPath $fbx -PathType Leaf) {
                    $corpusRoot = $child1.FullName
                    break
                }
            }
            foreach ($child2 in @(Get-ChildItem -LiteralPath $child1.FullName -Directory -ErrorAction SilentlyContinue)) {
                if ($child2.Name -eq $CorpusRootName) {
                    $fbx = Join-Path $child2.FullName (Join-Path $AssetRel "Mage.fbx")
                    if (Test-Path -LiteralPath $fbx -PathType Leaf) {
                        $corpusRoot = $child2.FullName
                        break
                    }
                }
            }
            if ($corpusRoot) { break }
        }
        if ($corpusRoot) { break }
    }
}
if (-not $corpusRoot) { throw "MAGE_OBSERVATION_REFRAME_CORPUS_ROOT_NOT_FOUND" }

$dst = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
$srcDst = Join-Path $dst "source"
New-Item -ItemType Directory -Path $srcDst -Force | Out-Null

$sourceDir = Join-Path $corpusRoot $AssetRel
Copy-Item -LiteralPath (Join-Path $sourceDir "Mage.fbx") -Destination (Join-Path $srcDst "Mage.fbx") -Force
Copy-Item -LiteralPath (Join-Path $sourceDir "mage_texture.png") -Destination (Join-Path $srcDst "mage_texture.png") -Force

$report = [ordered]@{
    schema = "RealSaS.MageObservationReframeWindowsStage.v2"
    status = "PASS__EXACT_NAMED_SOURCE_STAGED"
    recursive_file_discovery_used = $false
    bounded_named_root_search_used = $true
    source_fbx = "Mage.fbx"
    source_texture = "mage_texture.png"
}
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $dst "STAGE_REPORT.json") -Encoding UTF8
Write-Output "MAGE_OBSERVATION_REFRAME_STAGE_PASS"
Write-Output ("STAGE_ROOT=" + $dst)
