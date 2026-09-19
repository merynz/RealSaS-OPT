# RealSaS-OPT — Current State

**Date:** 2026-09-19  
**Canonical continuation/execution branch:** `main`  
**Active experiment gate:** `SUBJECT2_KNIGHT_FULL_CLOSURE` on `main`  
**Active run:** `SUBJECT2_KNIGHT_V1`  
**Progress:** **0/40** — next `01_SOURCE_BYTES_SEALED`  
**Current state:** `PRODUCT_AUTHORITY_IMPLEMENTATION__ALL_40_STAGES_EXECUTABLE__FINAL_CI_AND_RUN_INPUTS_PENDING`  
**Most recent closed gate:** `ALL_40_STAGE_ADAPTERS_IMPLEMENTED_AND_BOUND_READY_FOR_FINAL_CI`

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
All 40 canonical stages now have executable adapters and an explicit DAG. Stages 03–08 close source/admission/observation authority; stages 09–15 bind external IRIS fit receipts to exact zero-surface and Compiler GSA qualification; stages 16–23 bind Geppetto/Arachne proposal evidence to Compiler-owned QualifiedSkeletonIR/QualifiedSkinIR; stages 24–40 close product mesh, presentation, rest preservation, motion, Runtime-v4, native visual evidence and PRODUCT_PASS. The remaining work is final repository CI cleanup, canonical plan-hash refresh, and supplying the active Knight run's exact external fit/native-player artifacts before executing 01→40. No Knight PASS is claimed by implementation readiness alone.

## Hard boundaries
- Mage FIT/FIT2 remains historical/scoped evidence; it is not current execution authority.
- No `if subject == "Knight"` product code.
- Teacher/source rig is FIT evaluation evidence, never hidden shipping authority.
- Out-of-frame is UNKNOWN and cannot delete geometry.
- Canonical 3D geometry is never RGB appearance authority.
- Rest source preservation must pass before motion.
- Product closure requires visible native motion output, not JSON-only metrics.
