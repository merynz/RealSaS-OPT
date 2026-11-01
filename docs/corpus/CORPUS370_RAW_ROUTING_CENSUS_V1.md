# Corpus370 raw routing census

Status: **measurement only — no family or training authority**

This note records a raw-byte routing census over the exact 370-source bundle. It
exists to choose the correct producer/court lane per source format and authored
capability. Nothing below is a model-admission count.

## Exact candidate universe

| Origin | Format | Rows |
| --- | ---: | ---: |
| Established Truth139 | `.gltf` | 139 |
| Expansion | `.gltf` | 57 |
| Expansion | `.glb` | 63 |
| Expansion | `.fbx` | 111 |
| **Total** |  | **370** |

The raw source census re-parses all 370 immutable source bytes. It independently
reproduces the current Blender-inspection aggregate of **328 sources with authored
rig + skin evidence and 42 without it**. Authored animation structures are present
in 368 sources, but that does not mean 368 AXIS-ready subjects.

## Rig / skin routing

| Raw route | Rows |
| --- | ---: |
| `.gltf`: exact raw-source rig/skin + imported-equivalence lane already exists | 196 |
| `.glb`: exact GLB-container rig/skin + imported-equivalence adapter required | 21 |
| `.fbx`: pinned source import + independent source/replay equivalence lane required | 111 |
| No authored rig/skin detected: independent absence court required | 42 |

The 42 rows without authored rig/skin are the Kenney Cube Pets / Blocky Character
GLB lane. They are not automatically discarded from IRIS or graphics truth. They
simply cannot become MIRA skin or AXIS rig-program supervision without a separate
qualified target.

## Animation routing

| Raw route | Rows |
| --- | ---: |
| `.gltf`: raw authored animation + canonical replay lane already exists | 196 |
| `.glb`: raw GLB animation + canonical replay adapter required | 21 |
| `.fbx`: pinned source import + canonical replay lane required | 109 |
| Animation present but no authored rig/skin; node/morph evidence only, not AXIS rig kinematics | 42 |
| No authored animation detected; independent absence court required | 2 |

The two no-animation rows are the Universal Base Character FBX sources. They may
still carry useful geometry/appearance/rig/skin truth.

## Combined raw capability observation

- rig + skin + authored animation detected: **326**
- rig + skin but no authored animation detected: **2**
- animation container evidence but no authored rig/skin detected: **42**

These are work-routing observations, not MIRA/AXIS final roster counts.

## Lineage warning

There are no exact raw-source SHA duplicates among the 370 rows. That is necessary
but not sufficient for a leakage-safe split. At least 33 basename collision groups
exist across packs/exports (`Deer`, `Wizard`, `Yeti`, `Cow`, `Shark`, etc.). File
name is neither duplicate proof nor independence proof. Before the global unseen
split, lineage must be rebuilt from canonical structural fingerprints including
geometry, rig/skin and animation identity where available, with source-pack/export
relations retained explicitly.

## Consequence for the 370 closure

The old Full139 family sweep is valid as a reusable **glTF evidence route**, but it
cannot be copied blindly to the expansion cohort. The 370 campaign must dispatch
by source format and capability while producing the same normalized family claims.
Every resulting family claim still enters the central subject × family matrix only
after its current independent court passes.
