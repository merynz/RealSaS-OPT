# RealSaS-OPT — Current State

**Date:** 2026-09-16  
**Repository continuation branch:** `playback-stack-v1-20260916`  
**Parent product baseline:** `product/mage-fit1-demo-fulfillment-20260914` @ `e60ea9129f5576f3965f7e1b6b3d54d2e1437fe2`  
**Scientific FIT2 authority remains:** `fit2/mage-full-subject-reclosure`  
**Product goal:** `AUTOMATIC_8_DIRECTION_SPINE_CLASS_PUPPET_PLAYBACK`  
**Status:** `MESH_RIG_SKIN_AVAILABLE__PLAYBACK_STACK_ACTIVE__D0_CLOSED_PASS__D1_R0_R1_R4_IN_PROGRESS__FOUNDER_VISUAL_PASS_FALSE`

## Read first

1. `canonical/PLAYBACK_STACK_V1_20260916.md` — current playback/runtime continuation authority.
2. `canonical/FIT2_CANONICAL_EXECUTION_AUTHORITY_V1.json` — corrected FIT2 scientific execution authority.
3. `canonical/CONTEXT_STATE_V2.json`
4. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V2.md`
5. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V2.md`
6. `canonical/BRANCH_AUTHORITY_V2.md`

## One-line state

`Mesh, rig and skin no longer define product completion. RealSaS is now explicitly closing the generic post-rig Puppet Playback Stack: canonical 3D motion, 3D->2D deformation/projection, full-surface depth visibility, Spine-class slot/draw-order/clipping composition, source-provenance appearance, quality-controlled playback, native .rss runtime and C++ reference rendering. Mage is TEST_SUBJECT_001 only; FIT8/FITK/unseen must use the same contracts.`

## Hard product boundary

The current product stack is:

`Geometry -> Rigging -> Skin -> Motion -> Projection/Deformation -> Visibility/Composition -> Appearance -> Bake/Playback -> Native Runtime -> Reference Render -> PNG`

A PASS at one layer does not imply any downstream PASS. `PASS` without a named layer is forbidden in playback reporting.

## Current closed facts

- Corrected H1/GSA/Geppetto scientific lineage remains preserved under FIT2 authority.
- Mage demo uses sealed FIT1 product assets where explicitly authorized; this does not make the runtime Mage-specific.
- Dense/full zero-surface BODY visible-rest coverage is qualified; hidden BODY under rigid components is still unmeasured.
- ABC V2 established the selected-face visibility architecture as pathological; runtime-v3 direction is full-surface + posed depth visibility.
- D0 FIX1: **CLOSED PASS** — axis contract, neutral baseline and exact rest-deficit localization are frozen.
- D0 axis contract is geometry-derived; bind joint orientations/independent semantic run truth are not available.
- Appearance diagnostic flat/filled colors are not product texture authority.
- Historical ~98-99% rest coverage means foreground geometric coverage, not source-faithful texture and not correct visible layer.

## Explicit open state

- `BODY_VISIBLE_REST_COVERAGE = QUALIFIED`
- `BODY_UNDER_RIGID_SURFACE = UNMEASURED`
- `THIN_RIGID_COMPONENT_GEOMETRY_DEFICIT = OPEN_SEPARATE_CONCERN`
- `D1_MOTION_IMPLEMENTATION_PASS = NOT_CLAIMED`
- `R0_VISIBILITY_PASS = NOT_CLAIMED`
- `R1_RASTER_CONFORMANCE_PASS = NOT_CLAIMED`
- `R2_APPEARANCE_AUTHORITY_PASS = NOT_CLAIMED`
- `R3_PLAYBACK_PASS = NOT_CLAIMED`
- `R4_NATIVE_E2E_PASS = NOT_CLAIMED`
- `APPEARANCE_PRODUCT_PASS = FALSE`
- `FOUNDER_VISUAL_PASS = FALSE`

## Genericity rule

No new playback/runtime implementation may key behavior on Mage, FIT1, a Mage component name, Mage view statistics or Mage-specific topology. Subject-specific data is fixture input only. The active product target is automatic 8-direction playback for future FIT8, FITK and unseen subjects.

## External behavioral references

- **Spine runtime** is the primary behavioral reference for 2D runtime quality: slot/attachment state, animated draw order, clipping, deformation playback, order-preserving batching and performance discipline. RealSaS implementation is clean-room.
- **CharacterGen** is the primary clean-room reference for full reconstructed geometry + camera/depth visibility separation.

## Current code on continuation branch

- `compiler/realsas_compiler_core/playback_runtime_v3.py` — typed generic R0/R1/R2 runtime-v3 contract.
- `tests/compiler/test_playback_runtime_v3_contract_v1.py` — fail-closed contract tests.
- `experiments/playback_stack_v1/run_runtime_reference_e2e_v1.py` — generic `.rss -> native C++ renderer -> PNG` R4 probe.
- `canonical/PLAYBACK_STACK_V1_20260916.md` — complete architecture/continuation record.

Runtime-v2 remains readable compatibility state and is explicitly **no-depth**. It may be used only for structural smoke testing, never as depth-qualified/founder visual evidence.

## Immediate coding order

1. R1 native C++ reference-raster depth/conformance primitive + tests.
2. Runtime/package binary v3: posed z, slot/attachment state, clipping interval and explicit visibility policy, while keeping v1/v2 read compatibility.
3. Native C++ v3 depth renderer + compiler exporter.
4. D1 generic canonical 3D FK/LBS/projection qualification; save projected joint trajectories.
5. First real TEST_SUBJECT_001 R4 package/render.
6. R2-A source appearance + under-rigid audit; current Mage UNSEEN stays blank/marked.
7. D2 D-spine.
8. R2-B atlas baking + R3 playback qualification.
9. Native Founder sheet and only then `FOUNDER_VISUAL_PASS` decision.

## Current appearance policy

For TEST_SUBJECT_001, unseen regions are intentionally left `UNSEEN`; synthetic/nearest-color/generative completion is forbidden from product output for now. Future high-quality completion may be enabled only by explicit policy and must preserve provenance.

## Scientific continuity

FIT1 remains historical/scoped evidence where superseded. Corrected FIT2 scientific authority is not replaced by this playback branch. The playback branch consumes qualified product/scientific artifacts but owns the new generic post-rig product/runtime closure.
