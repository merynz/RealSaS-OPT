# RealSaS-OPT — Current State

**Date:** 2026-10-16
**Canonical code line:** `main` only
**Active program:** return to the proven Knight-derived architecture, remove only the IRIS→TESSA representation bottleneck, then resume/restart Fit8
**Deferred research:** broad Unified-IRIS / StdGEN++-inspired redesign, semantic-owner corpus/census, one-image synthesis, appearance/semantic/extraction expansion

## Executive state

The broad Unified IRIS redesign authorized on 2026-10-14 is now **DEFERRED by user decision**. It remains valuable research and must be preserved for later, but it is no longer the active architecture authority or a prerequisite for Fit8.

Draft PRs `#113` and `#114` are explicitly marked deferred and must not be merged into `main` during the current Fit8 restart program.

The project returns to the architecture that already produced the Knight animated demo:

```text
8-view source evidence
    ↓
IRIS
    ↓
TESSA learned topology proposal
    ↓
Compiler qualification / canonical M
    ↓
AXIS → canonical G
    ↓
MIRA → W_M on exact M
    ↓
Compiler mechanics / appearance / presentation / dynamic proof
    ↓
Runtime
```

AXIS, MIRA, CAA, presentation and runtime are **not reopened** by the current intervention.

## Preserved canonical continuity

This narrow refocus does **not** erase or replace the existing V2 control-plane/product authority records. The current machine-readable governance remains anchored by `canonical/V2_IMPLEMENTATION_READINESS.json`, the existing **43-stage** dependency DAG, `canonical/ACTIVE_RUN_V2.json`, and `canonical/AUTHORITY_MAP_V1.json`. The registered governance gate id `PLATFORM_EXECUTION_LANES` remains a control-plane continuity record on `main`; it is not a competing scientific architecture program.

The exact pre-existing governance context state token is `CARRIER_NATIVE_FIT1_SCOPED_PASS__PLATFORM_LANES_IMPLEMENTED`. It remains a continuity anchor for the V2 control plane; it does not override the narrower current scientific work order below.

The proven product concerns also remain first-class and explicit: **Geometry**, **Mechanics**, and **Appearance** are separate quality authorities; **QualifiedMeshIR** remains the typed qualified mesh authority entering the canonical carrier boundary; **SurfaceAddressing** remains the stable canonical-M addressing boundary; and **Complete Appearance Authority** / CAA remains the source-preserving appearance authority downstream of canonical M. Nothing in the IRIS→TESSA simplification weakens those contracts.

## The one pre-Fit8 architecture issue being fixed

The current IRIS→TESSA interface throws away rich learned scene information:

```text
IRIS image tokens / scene triplane
    ↓
continuous signed field
    ↓
deterministic tetra zero surface
    ↓
GSA / RiggingSurface
    ↓
17-D per-node TESSA conditioning
    ↓
second learned TESSA surface encoder
    ↓
TESSA topology token decoder
```

The 17-D row is exactly:

```text
XYZ(3)
normal XYZ + valid(4)
8 support-view bits
normalized graph degree
persistence bit
= 17
```

TESSA then projects this compressed representation back to a 768-D model and reconstructs a scene/surface memory with a separate learned encoder.

This is the active simplification target.

## Approved narrow target

Do **not** redesign the rest of RealSaS. Keep the proven IRIS signed-geometry model and the useful TESSA topology-token decoder, but remove TESSA as a second scene-understanding model.

Target logical model boundary:

```text
8 RGB
  ↓
existing IRIS scene encoder / triplane Z
  ├─ signed field → deterministic geometry witness/courts
  │
  └─ rich scene memory → bounded topology-memory adapter
                         ↓
                  existing TESSA topology decoder role
                         ↓
                 learned mesh proposal M*
```

The first implementation should consume the **existing IRIS scene planes directly** rather than modifying the core IRIS source merely to expose a new latent. This preserves compatibility with the existing Fit8 IRIS checkpoint lineage where possible.

The legacy `17-D → TESSASurfaceEncoderV1` path remains only as compatibility/reproduction code; it is not the intended Fit8 topology path after this intervention.

## Compiler authority around the simplified model

Model fusion does **not** mean Compiler authority moves into the network.

The learned topology proposal may be produced from IRIS scene memory before the geometry courts finish, but it remains non-authoritative evidence until the existing Compiler courts consume it.

Keep the current independent geometry/evidence chain:

```text
Stage08 normalization authority
        ↓
IRIS signed field
        ↓
Stage12 deterministic zero-surface witness
        ↓
Stage13 geometry substrate qualification
        ↓
Stage14/15 GSA build + qualification
        ↓
Stage17 mechanical/component partition evidence
```

In parallel, from the **same IRIS scene representation**:

```text
IRIS scene planes
        ↓
scene→topology memory adapter
        ↓
TESSA topology decoder role
        ↓
immutable learned proposal M*
```

Then the two paths meet at the existing Compiler carrier boundary:

```text
M* + qualified geometry/GSA/partition evidence
        ↓
Stage18 proposal binding + generic static repair + cross-evidence checks
        ↓
Stage19 exact canonical M
        ↓
AXIS → G
        ↓
MIRA → W_M
```

Important consequences:

- GSA remains valuable Compiler/mechanics evidence.
- Stage17 partition remains available for post-proposal support/binding.
- GSA/partition are no longer the sole learned information channel into topology prediction.
- `TESSAMaterialSupportFieldIR` remains a **post-proposal** material/mechanical support binding, not geometry authority and not the topology network input.
- Compiler static repair, Stage19 remeasurement, source-fidelity courts and exact carrier authority remain unchanged in principle.

This preserves the Knight authority chain while removing the neural information bottleneck.

## Model integration policy

The preferred first implementation is **logical IRIS+TESSA fusion with checkpoint-safe training**, not a broad end-to-end rewrite.

1. Keep `SceneFirstSignedGeometryV4` source/checkpoint semantics unchanged if possible.
2. Add a new topology-memory adapter that reads its existing `scene_planes` (`[B,3,64,64,64]` in the current Knight/Fit8 configuration).
3. Convert those rich plane features plus explicit plane/coordinate position into a bounded fixed topology memory.
4. Reuse the existing TESSA topology token grammar/decoder role.
5. Do not feed the topology head the 17-D GSA summary as its scene representation.
6. Preserve the old TESSA path for historical reproduction only.

A single wrapper/checkpoint may later contain IRIS + topology head, but the first Fit8 restart should preserve causal isolation:

- finish/resume IRIS geometry only if the stored checkpoint passes exact binding checks;
- train the shared topology adapter/decoder across all eight subjects from the sealed IRIS representation;
- keep IRIS frozen during the first topology-head fit unless a separately preregistered joint fine-tune is justified.

This removes the bottleneck without simultaneously changing the geometry learner that already reached the Fit8 run.

## Fit8 state

The previously stopped shared IRIS Fit8 run reached approximately global step **4400**. Preserve it and its exact code/input/policy bindings.

Because the active intervention now leaves the IRIS geometry model itself intact, that checkpoint is no longer rejected merely because of the deferred broad redesign. It may be resumed **only if** the existing resume guards prove exact dependency compatibility. Do not bypass or weaken those guards.

Fit8 remains:

- Snake
- Bat
- T-Rex
- Horse
- Spider
- GreenBlob
- Shark
- George

Knight remains a separate regression witness.

Fit8 proves shared-architecture/system closure, not unseen generalization.

## Explicitly deferred work

Do not put the following on the Fit8 critical path now:

- one-image → internally generated eight views;
- new appearance/RGBA head;
- generic semantic/ownership head;
- semantic owner cardinality census;
- amodal/occlusion teacher-corpus expansion;
- FlexiCubes-like extraction deformation/configuration head;
- large StdGEN-class capacity replacement;
- broad Unified IRIS teacher-corpus redesign;
- AXIS or MIRA redesign;
- new CAA/presentation architecture.

These remain research topics to revisit after Fit8 or when a specific measured failure requires them.

## Immediate work order

```text
1. keep deferred research PRs frozen
2. implement checkpoint-safe IRIS-scene → topology-memory adapter
3. reuse/refactor the existing TESSA token decoder without its 17-D scene encoder
4. add tests proving topology memory depends on rich IRIS scene planes and not GSA 17-D
5. preserve Stage12–19 Compiler authority/courts and adapt only the proposal import/binding contract where required
6. run CPU/self-hosted engineering tests
7. verify exact old Fit8 IRIS checkpoint compatibility
8. resume/restart shared Fit8 IRIS as allowed by the binding
9. train shared topology head across the same Fit8 cohort
10. continue to AXIS and MIRA only after canonical M closes
```

## Required read order for agents

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `AGENTS.md`
4. `SYSTEM_INDEX.md`
5. `docs/corpus/FIT8_ADMISSION.md`
6. `docs/platform/FIT8_PLATFORM_CAMPAIGN.md`
7. `canonical/CANONICAL_HANDOFF_20261007.md` for proven Knight mechanics provenance
8. deferred Unified-IRIS PRs/research only when explicitly revisiting that program

When the user says only `continue / devam et`, continue the narrow IRIS→TESSA bottleneck removal and Fit8 restart program. Do **not** silently reactivate the broad Unified-IRIS redesign.
