# Living Compile V4 Product Shell Promotion — 2026-09-05

## Decision

Promote the recovered V18.86/V18.87/V18.95 Living Compile product lineage into the current repository under `product/living_compile/`, rebound to the current V4 product/proof/runtime contracts.

This promotion does **not** restore the historical compiler, orchestrator, service authority, or superseded product schemas.

## Current authority boundary

- `compiler/realsas_compiler_core/` remains the only owner of canonical product truth.
- Living Compile consumes `RealSaS.CanonicalPuppetGraph.v3` and `RealSaS.ProductProofBundleIR.v1`.
- Proof-owned runtime preview consumes `RealSaS.QualificationOwnedMotionBakeIR.v1`; no second motion solver may be created by the UI/export shell.
- Directional display coordinates come from admitted raster bindings / qualified support, never by treating mechanical `P.xy` as raster coordinates.
- User edits are stored as `RealSaS.UserPuppetEditLayer.v3` beside the bundle, never inside the canonical bundle.
- Any user edit sets deployment status to `BLOCKED_PENDING_DYNAMIC_REVALIDATION` and requires Compiler requalification plus fresh dynamic proof before product/runtime promotion.

## Recovered editor surface

- rig pose rotation/translation;
- mesh vertex editing;
- weight editing surface;
- binding reassignment;
- eight-direction source/product inspection;
- textured triangle puppet preview;
- proof-owned runtime playback.

## Added authoring surface

- bone add/delete/reparent staging with cycle validation;
- IK constraint authoring with bounded validation;
- animation clip authoring;
- dope-sheet/keyframes;
- linear, step and cubic-smooth preview interpolation.

These authoring additions are user intent only. They do not claim that arbitrary edited topology/IK/animation has been qualified by the current Compiler/runtime lane.

## Export/preview bridge

`experiments/single_family_e2e_v1/export_bundle_v1.py` can now serialize exact proof-produced motion bake sidecars under `proof/motion_bakes/` when supplied by the E2E caller. The serializer rejects stale product bindings and never recomputes frames.

## Verification gate

`.github/workflows/living_compile_v4.yml` requires:

1. product-shell unit tests;
2. Python syntax validation;
3. JavaScript syntax validation.

The tests assert V4 lineage, raster-derived scene geometry, proof-owned runtime interpolation, sibling user-layer persistence, and canonical-bundle immutability.

## Non-claims

- This promotion does not claim one-family real FIT PASS.
- It does not claim generalization.
- Compile/training orchestration is not owned by the UI.
- User-authored edits are not deployable until a future/current Compiler requalification path explicitly accepts them and fresh proof passes.
