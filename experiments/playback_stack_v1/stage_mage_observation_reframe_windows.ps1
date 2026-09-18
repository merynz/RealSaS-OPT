param(
    [Parameter(Mandatory=$true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$H1RootName = "REALSAS_MAGE_FULL_SUBJECT_RECLOSURE_20260912"
$CorpusRootName = "RealSaS_MASTER_CORPUS_1024_V3"

function Candidate-Bases {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
        $root = [string]$drive.Root
        if (-not $root) { continue }
        foreach ($candidate in @(
            (Join-Path $root "My Drive"),
            (Join-Path $root "Google Drive\My Drive"),
            (Join-Path $root "Google Drive"),
            $root
        )) {
            if ((Test-Path -LiteralPath $candidate -PathType Container) -and -not $out.Contains($candidate)) {
                $out.Add($candidate)
            }
        }
    }
    return $out
}

$h1Root = $null
$corpusRoot = $null
foreach ($base in Candidate-Bases) {
    if (-not $h1Root) {
        $c = Join-Path $base $H1RootName
        if (Test-Path -LiteralPath (Join-Path $c "IRIS_H1_V2_INPUT\20260912_FULL_SUBJECT\V0.png") -PathType Leaf) {
            $h1Root = $c
        }
    }
    if (-not $corpusRoot) {
        $c = Join-Path $base $CorpusRootName
        $fbx = Join-Path $c "sources\gold\kaykit_adventurers\addons\kaykit_character_pack_adventures\Characters\fbx\Mage.fbx"
        if (Test-Path -LiteralPath $fbx -PathType Leaf) {
            $corpusRoot = $c
        }
    }
}
if (-not $h1Root) { throw "MAGE_OBSERVATION_REFRAME_H1_ROOT_NOT_FOUND" }
if (-not $corpusRoot) { throw "MAGE_OBSERVATION_REFRAME_CORPUS_ROOT_NOT_FOUND" }

$dst = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $dst) {
    Remove-Item -LiteralPath $dst -Recurse -Force
}
$obsDst = Join-Path $dst "current_observations"
$srcDst = Join-Path $dst "source"
New-Item -ItemType Directory -Path $obsDst,$srcDst -Force | Out-Null

$input = Join-Path $h1Root "IRIS_H1_V2_INPUT\20260912_FULL_SUBJECT"
for ($view=0; $view -lt 8; $view++) {
    foreach ($suffix in @(".png",".camera.json")) {
        $name = "V{0}{1}" -f $view,$suffix
        Copy-Item -LiteralPath (Join-Path $input $name) -Destination (Join-Path $obsDst $name) -Force
    }
}

$fbxSrc = Join-Path $corpusRoot "sources\gold\kaykit_adventurers\addons\kaykit_character_pack_adventures\Characters\fbx\Mage.fbx"
$texSrc = Join-Path $corpusRoot "sources\gold\kaykit_adventurers\addons\kaykit_character_pack_adventures\Characters\fbx\mage_texture.png"
Copy-Item -LiteralPath $fbxSrc -Destination (Join-Path $srcDst "Mage.fbx") -Force
Copy-Item -LiteralPath $texSrc -Destination (Join-Path $srcDst "mage_texture.png") -Force

$report = [ordered]@{
    schema = "RealSaS.MageObservationReframeWindowsStage.v1"
    status = "PASS__EXACT_NAMED_SOURCE_AND_CURRENT_OBSERVATIONS_STAGED"
    recursive_discovery_used = $false
    source_fbx = "Mage.fbx"
    source_texture = "mage_texture.png"
    current_observation_count = 8
    current_camera_count = 8
}
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $dst "STAGE_REPORT.json") -Encoding UTF8
Write-Output "MAGE_OBSERVATION_REFRAME_STAGE_PASS"
Write-Output ("STAGE_ROOT=" + $dst)
