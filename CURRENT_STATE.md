# RealSaS-OPT — Current State

**Date:** 2026-09-19  
**Canonical continuation/execution branch:** `main`  
**Active experiment gate:** `SUBJECT2_KNIGHT_FULL_CLOSURE` on `main`  
**Active run:** `SUBJECT2_KNIGHT_V1`  
**Progress:** **0/40** — next `01_SOURCE_BYTES_SEALED`  
**Current state:** `PRODUCT_AUTHORITY_IMPLEMENTATION__REST_PRESERVATION_EXECUTABLE__MOTION_SEAM_OPEN`  
**Most recent closed gate:** `REST_SOURCE_PRESERVATION_STAGE32_IMPLEMENTATION_PASS`

## Read first
1. `canonical/ACTIVE_RUN_V1.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V1.json`
3. `canonical/REAL_SAS_MAINLINE_VNEXT_20260918.md`
4. `canonical/AUTHORITY_MAP_V1.json`
5. this file

## Canonical architecture
`QualifiedCameraSetIR + 8 qualified observations -> IRIS dense signed geometry -> GSA/RiggingSurfaceIR S -> Geppetto/QualifiedSkeletonIR G -> Arachne/QualifiedSkinIR W -> MechanicalPartitionIR + boundary constraints -> view-independent MeshCandidate -> canonical QualifiedMeshIR M -> QualifiedMeshSkinIR B -> QualifiedPresentationGraphIR -> deterministic 8-view rest render -> rest source-preservation -> motion proof -> compact Runtime-v4 -> .rss -> native visual playback`

The architecture is subject-agnostic. **Knight is run data, never a branch in generic compiler/runtime code.**

## Execution rule
There is one 40-stage plan and one active-run ledger. A PASS stage is reusable only while its exact input fingerprint, implementation hash, policy hash and output SHA-256 identities still verify. A downstream failure does not erase upstream PASS artifacts. Upstream changes invalidate only the affected stage and descendants.

## Performance rule
The promoted runtime direction is the measured compact path: shared canonical XYZ once per asset per frame, static per-view overlays, streamed Runtime-v4 writer, content-addressed runtime/proof/render caches and native batch reference rendering. Historical synthetic evidence records an exact 8x XYZ reduction and 17.7x scalar-to-batch render speedup; this is optimization evidence, not a Knight or PRODUCT_PASS claim.

## Current task
Stage 05 and stages 24–32 now have executable typed authority adapters. Stage 32 replays the exact eight source rasters/foregrounds, independently measures rest preservation, re-derives the frozen subject-free policy, and fails if any view violates any hard rule. Its integration gate also closed a real half-pixel apparatus defect by separating source observation `PIXEL_CENTER_XY` texel coordinates from Runtime/G5 half-integer target centers without relaxing the frozen policy. The next scientific seam is Stage 33 motion-source/preset authority: it must bind the exact qualified rest-preservation lineage before Stage 34 may compile run/idle motion.

## Hard boundaries
- Mage FIT/FIT2 remains historical/scoped evidence; it is not current execution authority.
- No `if subject == "Knight"` product code.
- Teacher/source rig is FIT evaluation evidence, never hidden shipping authority.
- Out-of-frame is UNKNOWN and cannot delete geometry.
- Canonical 3D geometry is never RGB appearance authority.
- Rest source preservation must pass before motion.
- Product closure requires visible native motion output, not JSON-only metrics.
