# RealSaS — Post-Corpus Stage-A Closure Decision V2

**Date:** 2026-08-23  
**Build:** `REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822`  
**Status:** `STAGE_A_COMPLETE__GATE1_PASS_WITH_ONE_QUARANTINE__GATE2_3_SELECTIVE_AUDIT_REQUIRED`

## Gate 1 decision

- selected canonical assets: **3993**
- `primary_geometry.npz`: **3993/3993**, 0 missing, 0 duplicate, 0 zero-size
- render folders: **3993/3993**
- V0..V7 folders: **31,944/31,944**
- raster authorities: **31,944/31,944**
- exact all-eight-view blank assets: **1**
- blank asset: `asset_7da4136e89f94d4ce0d18337`

The earlier `primary_geometry missing=2398` value is formally falsified as an auditor timestamp-filter artifact.

The single blank asset is a localized admission/observation-coverage failure. It must be quarantined from IRIS observation training or marked IRIS-ineligible. It does **not** authorize a global rerender.

## Gate 2 decision

All selected assets have parsed technical authority and pass the builder's geometry/canonicalization gate. Source-container census among selected assets:

- `.blend`: **385** — evaluated-depsgraph / triangulation / authored-normal risk subset
- `.fbx`: **186**
- `.obj`: **64**
- `.glb`: **8**
- no locally preserved raw-source path: **3350**

A deterministic 64-asset `.blend` sample was frozen for selective Blender authority probing. Gate 2 remains open only for this target-authority test; expand beyond the sample only if failure prevalence is material.

## Gate 3 decision

Important scope correction: `raw_source_missing=0` means **0 missing among assets that have a preserved raw-source path**, not 0 missing among all 3993.

Locally preserved raw-source coverage is **643/3993**:

- KayKit: 37/37 present
- Objaverse linked/local subset: 415/415 present
- Quaternius: 191/191 present

The remaining **3350/3993** selected assets have no local preserved raw-source path in the current master corpus. Their appearance recovery therefore requires source-reference / URL re-materialization qualification before claiming an appearance-preserving B pass can cover them.

Container-class census:

- no local raw path / `OTHER__INSPECT`: **3350**
- FBX embedded-or-external unknown: **186**
- BLEND packed-or-external unknown: **385**
- OBJ MTL/texture external risk: **64**
- GLB embedded appearance possible: **8**

A 56-asset appearance inspection sample is frozen for Stage-B. This is not sufficient to close the 3350 linked-original recoverability question; that remains a separate Gate-3 item.

## Provenance / duplicate decision

- admission JSON parsed: **3993/3993**
- technical JSON parsed: **3993/3993**
- duplicate selected source-SHA groups: **0**
- source-provider imbalance warning remains active.

## Next legal execution

1. Stage-B selective Blender/source-content audit on the frozen samples.
2. If Stage-B shows material evaluated-mesh or normal-authority divergence, quantify/repair only the affected class/subset.
3. Independently qualify re-materialization of the 3350 non-local linked originals for appearance B coverage.
4. Freeze Gate-2/3 target/observation authority.
5. Only then run the preregistered exact Representation Authority Study.

No IRIS optimizer step is authorized by Stage-A.
