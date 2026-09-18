# RealSaS-OPT Agent Entry Contract

This repository must be resumable without conversational memory.

## Mandatory first read
1. `canonical/ACTIVE_RUN_V1.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V1.json`
3. `CURRENT_STATE.md`
4. `canonical/AUTHORITY_MAP_V1.json`
5. `canonical/REAL_SAS_MAINLINE_VNEXT_20260918.md`

## Resume rule
Read `canonical/ACTIVE_RUN_V1.json`. Verify `pipeline_plan_sha256` against the canonical plan. Find the first stage whose status is not `PASS` or `CACHE_HIT`; continue there. Never rerun a verified upstream stage merely because a later stage failed.

## Branch rule
`main` is the sole current continuation and execution branch. Historical branches are evidence stores. Do not create subject continuation branches.

## Genericity rule
Mainline compiler/runtime/orchestrator code must not branch on Mage, Knight, filenames, topology counts or subject IDs. Subject-specific facts belong in exact source/run manifests and sealed artifacts only.

## Ownership
- IRIS proposes observation-conditioned dense geometry evidence.
- Compiler/GSA owns admitted geometry and `RiggingSurfaceIR`.
- Geppetto proposes skeleton evidence; Compiler owns legal `QualifiedSkeletonIR`.
- Arachne proposes skin evidence; Compiler owns legal `QualifiedSkinIR`.
- `DrawableSurfaceIR` derives from the same dense lineage and binds explicitly to `RiggingSurfaceIR`.
- Runtime consumes qualified identities; it cannot create hidden geometry/rig/skin/appearance truth.

## Failure and cache discipline
Every stage records input fingerprint, implementation hash, policy hash, output SHA-256 identities and blockers in the active-run ledger. Resume is fail-closed: stale or missing outputs invalidate that stage and downstream only. No `latest` aliases are authority.

## Performance discipline
Prefer the promoted compact Runtime-v4 path and exact caches. Multi-node visual proof should use native batch rendering. Optimization must preserve byte/typed identity unless a separately preregistered lossy representation is qualified.

## Observation / appearance / science
Controlled FIT inputs require the full admitted subject in all eight views with safety margin. OOF is UNKNOWN. 3D carries mechanics/visibility, never RGB authority. Preserve old PASS artifacts with their exact scope; never threshold-launder or widen a local result into product/generalization authority.
