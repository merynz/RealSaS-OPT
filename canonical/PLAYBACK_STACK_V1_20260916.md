# RealSaS Playback Stack V1 — Continuation Authority

**Date:** 2026-09-16  
**Active continuation branch:** `playback-stack-v1-20260916`  
**Parent product baseline:** `product/mage-fit1-demo-fulfillment-20260914` @ `e60ea9129f5576f3965f7e1b6b3d54d2e1437fe2`  
**Product goal:** `AUTOMATIC_8_DIRECTION_SPINE_CLASS_PUPPET_PLAYBACK`

This document is the canonical continuation note for the post mesh/rig/skin playback phase. It exists so a new chat/agent does not reconstruct the plan from scattered experiment notes.

## Non-negotiable scope

The playback pipeline is **generic RealSaS product architecture**. Mage is only `TEST_SUBJECT_001`. No runtime, renderer, motion, visibility, appearance, atlas, bake, or export rule may be Mage-specific. FIT8, FITK and unseen subjects must enter the same typed contracts.

## External behavioral references

- **Spine runtime:** primary behavioral reference for 2D runtime architecture, slot/attachment state, animated draw order, clipping semantics, deformation playback, order-preserving batching and performance discipline. Source may be inspected for behavior, but RealSaS implementation remains clean-room and does not copy Spine runtime code.
- **CharacterGen:** primary clean-room reference for the geometry/camera/visibility separation: preserve the full reconstructed surface, transform/project the surface, and resolve visibility by depth rather than destructive rest-pose face selection.

These are references, not product dependencies.

## Product stack boundary

Mesh / Rig / Skin do not imply a rendered moving PNG. The product stack is:

`Geometry -> Rigging -> Skin -> Motion -> Projection/Deformation -> Visibility/Composition -> Appearance -> Bake/Playback -> Native Runtime -> Reference Render -> PNG`

A PASS at one layer never implies a downstream PASS.

## Current scientific facts

### Visibility diagnosis

ABC V2 established that the old selected-face construction is structurally pathological. A large fraction of selected faces are not rest z-buffer first-hit faces and selection is highly fragmented. The architecture direction is therefore:

`FULL_SURFACE_AUTHORITY -> POSE -> CAMERA -> DEPTH_VISIBILITY`

Visibility by destructive face deletion is forbidden in runtime-v3.

### D0 motion-axis closure

D0 FIX1 closed the generic axis/baseline precondition:

- `D0_AXIS_CONTRACT = CLOSED_PASS`
- `D0_NEUTRAL_BASELINE = CLOSED_PASS`
- `D0_REST_DEFICIT_LOCALIZATION = CLOSED_PASS`
- `BODY_VISIBLE_REST_COVERAGE = QUALIFIED`
- `BODY_UNDER_RIGID_SURFACE = UNMEASURED`
- `THIN/RIGID_COMPONENT_GEOMETRY_DEFICIT = OPEN_SEPARATE_CONCERN`
- `SEMANTIC_MOTION_TRUTH = NOT_AVAILABLE`
- `APPEARANCE_PRODUCT_PASS = FALSE`
- `FOUNDER_VISUAL_PASS = FALSE`

The ~98-99% historical rest coverage numbers are geometric foreground coverage metrics, not texture/appearance quality and not correct-layer guarantees.

## Appearance policy for TEST_SUBJECT_001

For the current Mage test, unseen appearance remains visibly **UNSEEN**. No nearest-color or generative/completion result may be presented as product appearance. Future completion is allowed only after an explicit product-policy decision and must retain provenance.

Canonical provenance classes:

- `DIRECT_SOURCE`
- `OTHER_VIEW_SOURCE`
- `UNDER_RIGID_SOURCE`
- `UNSEEN`
- `COMPLETION` — disabled by current product policy

Appearance donor authority is face/patch coherent, not arbitrary per-vertex donor mixing.

## Contracts

### D1 — Canonical 3D motion implementation qualification

Required before D2:

- frozen D0 rest axes transported by hierarchy: `axis_world = R_parent @ axis_rest`;
- fixed rotation composition order;
- zero-rotation FK identity;
- zero-rotation LBS identity;
- 3D bone-length preservation;
- truly independent dual LBS implementations (matrix path vs quaternion+translation path);
- sagittal side-view parity against legacy 2D authored intent, with tolerance locked before results;
- opposite side-view mirror witness;
- front/back near-zero lateral drift for sagittal terms;
- knee/elbow hyperextension guard;
- half-period bilateral limb reflection;
- loop endpoint position **and velocity** continuity;
- projected joint trajectories persisted per view for D2;
- no per-frame affine re-fit;
- depth taken from the same posed 3D state;
- twist is undefined without bind orientation and is not represented by D-spine V1.

Implementation parity is not semantic-motion truth.

### R0 — Runtime visibility contract

- full-surface geometry survives compilation;
- posed per-vertex depth is available to runtime/reference renderer;
- deformable BODY self-occlusion uses depth test, not mesh draw order;
- BODY writes depth;
- rigid/component attachments depth-test and use explicit alpha/depth-write policy;
- semantic slot/draw order remains for 2D composition and depth ties;
- clipping is an explicit draw-order interval semantic, not an ad-hoc image mask;
- visibility by deleting a rest-selected face subset is forbidden.

### R1 — Single reference raster contract

The C++ reference renderer owns product raster conformance. Notebook/nvdiffrast and UI preview are diagnostic implementations and must conform to the same contract.

Locked fields:

- half-integer pixel center;
- top-left triangle fill rule;
- top-left UV origin;
- bilinear sampling;
- clamp-to-edge;
- sRGB RGBA8;
- straight alpha + source-over;
- posed camera-forward depth, smaller is nearer;
- deterministic semantic depth tie rule;
- explicit winding/culling policy;
- no per-frame affine fit.

### R2 — Appearance authority

R2-A audit can proceed before D2; R2-B final atlas baking follows posed visibility.

- donor selection is face/patch coherent;
- donor must be visibility-qualified, not smallest-index fallback;
- cross-view texels are compile-time baked into the target runtime atlas;
- provenance remains inspectable;
- current Mage `UNSEEN` stays blank/marked;
- `COMPLETION` is opt-in and currently disabled;
- under-rigid BODY is measured separately via depth peeling/second-layer evidence.

### R3 — Bake/playback

- nine uniform samples are not final quality authority;
- bake density is quality-controlled (initial product target: sufficiently dense playback, expected ~30 fps or error-adaptive equivalent);
- avoid visible chord shortening/jitter from sparse vertex lerp;
- loop position and velocity continuity are qualified;
- crossfade semantics must preserve visibility/composition correctness;
- clipping interval transport exists in runtime-v3, but native clipping/stencil execution is **not yet qualified** and must fail closed.

### R4 — End-to-end product skeleton

Required path:

`compiled subject -> .rss -> native C++ runtime -> C++ reference renderer -> PNG`

Founder sheets are generated from this path, never from notebook-only diagnostic renderers.

Runtime-v2/no-depth may be exercised only as `STRUCTURAL_SMOKE_ONLY`; it cannot claim depth-qualified visual/product PASS.

## Spine-class runtime semantics adopted as RealSaS concepts

- shared compiled puppet asset vs lightweight per-instance state;
- Bone -> Slot -> active Attachment separation;
- draw order is explicit runtime pose/composition state;
- clipping begins/ends over a draw-order interval;
- clipping may change generated triangle topology, therefore immutable-topology fast paths are conditional;
- batching preserves draw order and only merges contiguous compatible submissions;
- compiler may classify topology as static, attachment-dynamic, clip-dynamic or draw-order-dynamic.

## Current code landed on active branch

1. `compiler/realsas_compiler_core/playback_runtime_v3.py`
   - generic typed runtime-v3 contract;
   - per-view mesh variants for logical attachments;
   - full-surface/depth requirements;
   - slots, attachments, clipping intervals;
   - face/patch appearance provenance;
   - explicit current completion prohibition;
   - deterministic R1 raster contract.
2. `compiler/realsas_compiler_services/export/runtime_v3.py`
   - `.rss`/`.rsr` v3 writer;
   - rest `XYZUV`;
   - posed per-frame `XYZ`;
   - per-view meshes, slot draw order, active attachments and clipping interval transport;
   - source/proof/contract hashes and render-capability manifest.
3. `runtime/realsas_cpp/src/reference_raster_v3.h`
   - half-pixel/top-left coverage;
   - winding-independent barycentric identity;
   - posed-depth compare;
   - semantic equal-depth tie;
   - explicit depth-write/cutout policy.
4. `runtime/realsas_cpp/src/runtime_v3_reference.{h,cpp}`
   - native v3 binary/ZIP reader;
   - binary and texture CRC validation;
   - slot/active-attachment playback;
   - XYZ interpolation;
   - posed depth-buffer reference raster;
   - clipping fail-closed until native clipping qualification.
5. `runtime/realsas_cpp/examples/realsas_runtime_v3_demo.cpp`
   - real `.rss v3 -> native C++ -> PNG` executable.
6. `runtime/realsas_cpp/tests/runtime_v3_writer_native_e2e.py`
   - exact production writer -> native reader/renderer test;
   - adverse painter order witness requires near/green surface to win by depth.
7. `tests/compiler/test_playback_runtime_v3_contract_v1.py`
   - fail-closed contract tests.
8. `experiments/playback_stack_v1/run_runtime_reference_e2e_v1.py`
   - generic `.rss -> native C++ renderer -> PNG` R4 probe;
   - runtime-v2 is explicitly structural smoke only.

## Execution evidence recorded on 2026-09-16

- `R1_NATIVE_REFERENCE_TRANSLATION_UNIT_COMPILE = PASS` under local GCC C++17 with libarchive/libpng/zlib.
- `R1_NATIVE_REFERENCE_LINK = PASS`.
- `R1_SYNTHETIC_POSED_DEPTH_BEHAVIOR = PASS`.
  - synthetic v3 package intentionally submitted the near/green surface before the far/red surface;
  - a painter-only renderer would end red;
  - native output pixel was exactly `(0,255,0,255)`;
  - therefore posed depth, not submission order, owned the result.
- `R4_SYNTHETIC_RUNTIME_V3_NATIVE_DEPTH_E2E = PASS`.
- `R4_REAL_PRODUCT_V3_E2E = OPEN`.
- `R3_NATIVE_CLIPPING = OPEN_FAIL_CLOSED`.
- `FOUNDER_VISUAL_PASS = FALSE`.

The synthetic E2E proves the format/parser/depth-render execution path. It does **not** promote Mage or any subject to product visual pass.

## Immediate next coding order

1. Implement native Spine-class clipping interval/stencil semantics and conformance tests (`R3_NATIVE_CLIPPING`).
2. Implement the **real product -> runtime-v3 projector** from current proof-owned full-surface/appearance/motion authority; no selected-face fallback.
3. Wire TEST_SUBJECT_001 through `.rss v3 -> realsas_runtime_v3_demo -> V0..V7 PNG` and inspect real output.
4. Implement D1 generic FK/LBS/projection qualification and persist per-view joint trajectories where current motion authority still lacks canonical 3D posed depth.
5. Run R2-A appearance/under-rigid audit with `UNSEEN` unfilled.
6. Implement D2 D-spine using the frozen D1/R0/R1 contracts.
7. R2-B atlas bake, R3 playback density/crossfade qualification, then Founder Visual Pass.

## PASS language rule

Every PASS must name its layer. `PASS` without a layer is forbidden in playback reports.

Examples:

- `D1_MOTION_IMPLEMENTATION_PASS`
- `R0_VISIBILITY_PASS`
- `R1_RASTER_CONFORMANCE_PASS`
- `R2_APPEARANCE_AUTHORITY_PASS`
- `R3_PLAYBACK_PASS`
- `R4_NATIVE_E2E_PASS`
- `FOUNDER_VISUAL_PASS`

Until the final native path renders source-faithful animated output, `FOUNDER_VISUAL_PASS = FALSE`.
