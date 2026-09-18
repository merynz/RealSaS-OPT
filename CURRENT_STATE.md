# RealSaS-OPT — Current State

**Date:** 2026-09-18  
**Canonical continuation/execution branch:** `main`  
**Active experiment gate:** `SUBJECT2_KNIGHT_FULL_CLOSURE` on `main`  
**Active run:** `SUBJECT2_KNIGHT_V1`  
**Progress:** **0/40** — next `01_SOURCE_BYTES_SEALED`  
**Current state:** `PRODUCT_AUTHORITY_IMPLEMENTATION__QUALIFIED_MESH_ADMISSION_ACTIVE`  
**Most recent closed gate:** `GENERIC_RUNTIME_OPTIMIZATION_SPINE_PROMOTED`

## Read first
1. `canonical/ACTIVE_RUN_V1.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V1.json`
3. `canonical/REAL_SAS_MAINLINE_VNEXT_20260918.md`
4. `canonical/AUTHORITY_MAP_V1.json`
5. this file

## Canonical architecture
`8 qualified observations -> IRIS dense signed geometry -> GSA/RiggingSurfaceIR S -> Geppetto/QualifiedSkeletonIR G -> Arachne/QualifiedSkinIR W -> MechanicalPartitionIR + boundary constraints -> view-independent MeshCandidate -> canonical QualifiedMeshIR M -> QualifiedMeshSkinIR B -> QualifiedPresentationGraphIR -> rest source-preservation -> motion proof -> compact Runtime-v4 -> .rss -> native visual playback`

The architecture is subject-agnostic. **Knight is run data, never a branch in generic compiler/runtime code.**

## Execution rule
There is one 40-stage plan and one active-run ledger. A PASS stage is reusable only while its exact input fingerprint, implementation hash, policy hash and output SHA-256 identities still verify. A downstream failure does not erase upstream PASS artifacts. Upstream changes invalidate only the affected stage and descendants.

## Performance rule
The promoted runtime direction is the measured compact path: shared canonical XYZ once per asset per frame, static per-view overlays, streamed Runtime-v4 writer, content-addressed runtime/proof/render caches and native batch reference rendering. Historical synthetic evidence records an exact 8x XYZ reduction and 17.7x scalar-to-batch render speedup; this is optimization evidence, not a Knight or PRODUCT_PASS claim.

## Current task
Finish G1/G5 subject-free policy preregistration and bind stages 24–30 to the implemented authority adapters. QualifiedMesh now has deterministic candidate promotion, intrinsic G1/G2/G4 recomputation, the subject-free 7.5° / aspect-16 G3 numerical floor, typed carrier policy and typed mesh qualification policy. No Knight mesh result has been inspected for threshold selection.

## Hard boundaries
- Mage FIT/FIT2 remains historical/scoped evidence; it is not current execution authority.
- No `if subject == "Knight"` product code.
- Teacher/source rig is FIT evaluation evidence, never hidden shipping authority.
- Out-of-frame is UNKNOWN and cannot delete geometry.
- Canonical 3D geometry is never RGB appearance authority.
- Rest source preservation must pass before motion.
- Product closure requires visible native motion output, not JSON-only metrics.
