# RealSaS-OPT — Current State

**Date:** 2026-10-18  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Verified pre-docs code head:** `b38feee7f8fafa2308b8e3c9cecdc22f248e28a7`  
**Verified workflow:** `39126288809` = `completed/success`  
**Review vehicle:** draft PR #116; currently stale relative to the active source line  
**Active program:** **POST-V8 QUALITY / TRUTH / CONSUMPTION QUALIFICATION**  
**Scientific status:** **NO IRIS FIT8 SCIENTIFIC PASS**

## Authority separation

Do not collapse these into one notion of “current”:

1. `main` is merged production-code authority.
2. The completed V8 Fit8 run is the architectural/measurement baseline, bound to source commit `5083c9915533ec8ec6530c7d229c635887abfea7` plus exact external checkpoint/receipt lineage.
3. The active post-V8 branch contains future-source hardening and audits. It does not retroactively mutate V8.
4. Artifact Registry / Attempt / ProductRevision remain durable research/product authority. Git chronology is not a substitute.

## Training lineage

### V5 — downstream single-objective rewriting falsified

A1 improved appearance while mechanics/surface authority collapsed. A downstream objective can improve its own metric while rewriting useful shared representation.

### V6 — cumulative retention only partially worked

Prior-objective retention preserved mechanics better but geometry/extraction still drifted because post-M1 `scene_core` stayed trainable (`scene_lr_scale=0.25`). Loss balancing alone was insufficient carrier protection.

### V7 — repair only

V7 repaired the TESSA witness-coordinate contract. It did not change V6 curriculum weights, optimizer policy or carrier-update policy.

### V8 — post-M1 carrier isolation supported

V8 migrated from the exact V6 M1-end state, sealed `scene_core`, reached step12000, and held geometry/extraction invariant while appearance/relation/topology objectives continued learning.

Current default invariant:

```text
G / M establish shared carrier H
        ↓
M1 seal
        ↓
A / R / T / J specialist learning
        ↓
scene_core remains sealed
```

Historical V8 already had the carrier seal and stage-level specialist isolation. Its physical AdamW groups were `scene_core`, combined `evidence_heads` (mechanical + appearance + relations), and `structure`. Post-V8 `optimization_v2.py` does **not** invent the seal; it separates future specialist experiments into `mechanical_head`, `appearance_head`, `relation_head`, and `structure` groups so LR, gradient and update telemetry are independently attributable while `scene_core` remains LR=0 fail-closed.

V8 is a **measurement baseline**, not a scientific Fit8 PASS: metric-specific preregistered thresholds and an independent topology court were absent.

## Executable carrier hardening

`models/iris/vnext/quality_policy_v1.py` uses exact canonical post-M1 stage IDs:

```text
A1_APPEARANCE_EVIDENCE
R1_ROLE_FREE_RELATIONS
T1_MECHANICAL_STRUCTURE
J1_JOINT_FINETUNE
```

Unknown A/R/T/J aliases fail closed. Future V8-line trainer execution checks both configured `scene_lr_scale` and the actual optimizer `scene_core` group LR.

`models/iris/vnext/optimization_v2.py` is for future focused post-V8 experiments only:

```text
scene_core       LR = 0, fail closed
mechanical_head  independent LR / telemetry
appearance_head  independent LR / telemetry
relation_head    independent LR / telemetry
structure        independent LR / telemetry
```

No specialist LR defaults are silently invented. Historical V8 optimizer provenance remains unchanged.

## Current architecture

```text
observations + cameras
        ↓
      IRIS
 shared scene carrier H
   ┌────┼──────────────┐
   ↓    ↓              ↓
Geometry Mechanics  specialist evidence
 field   demand      appearance / relations / structure
   └────┬──────────────┘
        ↓
  Evidence Package
        ↓
     Compiler
   ┌────┼─────────┐
   ↓    ↓         ↓
fragments order contact/attachment
        ↓
 one canonical M/G/W_M
        ↓
     Runtime
```

No mandatory named body-part taxonomy exists in the inference contract. No independent deformable visual mesh is allowed.

## Relation truth / conditioning state

The V8 audit found projected alpha/appearance targets derived from render views while `GraphicsRelationHeadV1.static_relations(Ha,Hb)` predicted them without camera conditioning. This is a supervision/conditioning mismatch, not merely an LR problem.

Replacement source is implemented and green:

- `models/iris/vnext/graphics_truth_v2.py` — authority/provenance/validity;
- `models/iris/vnext/training_v2.py` — validity-mask loss;
- `tools/iris_vnext/build_fit8_relation_truth_v2.py` — versioned sidecar/manifest producer;
- `models/iris/vnext/relation_support_v2.py` — support/class-balance court eligibility;
- `models/iris/vnext/heads_v2.py` — projected boundaries moved to the camera-conditioned branch with deterministic V1→V2 migration semantics;
- `models/iris/vnext/optimization_v2.py` — head-separated focused optimizer and convergence telemetry;
- focused CI is green at `b38feee7...`, run `39126288809`.

What is **not** established:

- no repository evidence seals a materialized Fit8 Relation Truth V2 corpus/manifest as experiment authority;
- no real V8 checkpoint→RelationHeadV2 migrated checkpoint/receipt is sealed;
- no focused post-V8 relation/boundary optimization has run;
- no new scientific relation PASS exists.

Static `same_drawable`, `cut`, and `attachment` abstain in Truth V2 when independent authority is absent. Current deterministic `occlusion_probability` support is single-class on its valid two-hit rows; it may be training/measurement evidence but is not eligible for a binary held-out PASS court until authoritative negative support exists. Do not fabricate negatives.

## Boundary causality

`BoundaryCausalityTeacherV1` admits only source-defensible causes:

```text
GEOMETRIC_TERMINATION
MATERIAL_BOUNDARY
ALPHA_BOUNDARY
OCCLUSION_VISIBILITY_BOUNDARY
SOURCE_PRIMITIVE_COMPONENT_SEAM
RESIDUAL_UNKNOWN
```

`RESIDUAL_UNKNOWN` is abstention and cannot become positive training evidence.

## Foundation branch disposition and salvage map

Research provenance branch:

`agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`

Source-defensible tools to inspect/port selectively:

- geometry/topology: `tools/corpus/qualify_f_geometry_carrier_v1.py`, `fit8_surface_topology_invariants_v1.py`;
- render↔geometric + glTF material/skin provenance: `fit8_render_vertex_equivalence_v1.py`, `fit8_gltf_static_carrier_v1.py`;
- FBX exact-source census: `ufbx_static_census_probe_v2.c`, `ufbx_static_census_probe_v3.c`, `fit8_fbx_static_carrier_v1.py`, `fit8_fbx_static_carrier_v2.py`;
- static all-hit evidence: `fit8_static_ray_census_v1.py`, `fit8_static_ray_census_v2.py`, `fit8_static_ray_census_v3.py`;
- authored-motion fixed-frame evidence: `fit8_gltf_dynamic_pose_v1.py`, `fit8_dynamic_fixed_frame_presentation_v1.py`, `fit8_dynamic_mechanics_metrics_v1.py`, `fit8_gltf_dynamic_census_v2.py`;
- residual boundary: `fit8_f_residual_boundary_graph_v1.py`, `fit8_f_residual_signature_v1.py`, `fit8_face_strand_boundary_incidence_v1.py`;
- local continuation: `fit8_local_patch_teacher_v1.py`, `fit8_patch_face_support_v1.py`, `fit8_f_patch_continuation_court_v1.py`, `fit8_f_face_qualified_continuation_court_v1.py`.

Do not promote authoritative drawable ownership, hidden/amodal support, intended 2D presentation override, global semantic ontology, or inferred hidden continuation from generic FIT8 3D alone.

## Compiler/runtime consumption

Machine-readable audit: `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Current source evidence supports fragment clustering from same/cut/boundary, directed order from front+occlusion, explicit native precedence overriding contradictory physical depth, and visibility changing final native RGBA. Attachment records survive into `graphics.bin` but the current native reference raster does not behaviorally use them. `support_band_pixels` survive serialization/validation but are not applied by the reference renderer.

Serialized is not the same as behaviorally used.

## TESSA/topology state

V8 showed topology-objective learnability from sealed H but did not seal an independent topology PASS. Required held-out court must measure actual topology/surface outcomes: valid/degenerate faces, component error, boundary loops/holes, non-manifold behavior where meaningful, canonical-surface deviation, source support and held-out stability.

Training loss is not topology qualification.

## StdGEN++ re-audit

Detailed dossier: `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`.

Pinned public code inspected:

- StdGEN `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766`;
- InstantMesh `TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.

The public StdGEN repository does not expose the full training loop, so paper-only optimizer/curriculum details remain `PAPER_EXACT`, not `CODE_EXACT`.

Transferable discipline: protect a useful carrier, constrain specialization, preserve earned supervision, require special data for hidden/internal truth, and consider coarse-to-fine extraction/topology regularization only after a held-out court identifies the relevant failure. Generic LoRA/adapters remain unauthorized until a clean-target capacity court fails.

## Self-hosted stale-run incident — 2026-10-18

Self-hosted run `39121291171` / `iris-vnext-fit8-prepare-self-hosted-v1 #22` was queued on stale PR #116 head `research/iris-vnext-single-carrier-graphics@90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6` and started when `realsas-wsl-1660ti` came online.

The stale run locally produced:

```text
TEACHER_BATCH_RECEIPT sha256 = 7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27
COHORT manifest value sha256 = 37a5fccbdcb2d98735291c7963e75e852e260e4d4eecfc2ef9f60ff8dbcb732b
PREPARATION_RECEIPT sha256 = fdd72430a9e6573b51b98eddf181d159cca08079654ceed08de92a25ae840082
source_commit = 90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6
```

It then entered Drive persistence to `gdrive:RealSaS_Fit8_IRIS_vNext_20261016`. The run was cancelled during that step before round-trip SHA verification completed. GitHub reports run conclusion `cancelled`; a compact forensic artifact was retained as `iris-vnext-fit8-preparation-39121291171` (artifact id `12137271848`).

Therefore:

- the stale run is **not** current preparation authority;
- the remote Drive preparation path is **QUARANTINED / UNVERIFIED** until re-audited against exact expected SHA/provenance;
- no current training may consume that remote path merely because files exist;
- runner remains offline until the current branch owns a safe verification/preparation workflow.

## Quality / PASS discipline

`0.1–0.2` residuals are not normalized into success because training finished. `~0.05` is only historical planning context for genuinely comparable normalized residual families, not a global threshold.

```text
MEASUREMENT_ONLY
OPTIMIZATION_DEBT
ELIGIBLE_FOR_COURT
COURT_PASS
```

`COURT_PASS` requires authoritative target + held-out metric + preregistered metric-specific threshold + required non-regression.

## Immediate next execution tranche

1. keep stale Drive preparation quarantined and verify its exact file hashes/provenance before reuse or replacement;
2. materialize Relation Truth V2 sidecars/manifest against exact admitted Fit8 inputs and seal provenance;
3. run per-family support/class/abstention audit;
4. create a real V8 relation-head state → `GraphicsRelationHeadV2` migration checkpoint/receipt;
5. preregister and run focused relation/boundary optimization with `scene_core` frozen and head-specific telemetry;
6. build the independent TESSA topology court;
7. close remaining evidence→Compiler→runtime `DEFINED → PRODUCED → SERIALIZED → CONSUMED → BEHAVIORALLY_USED` gaps;
8. synchronize PR #116 only after the final green review head is coherent.

No broad 12k/A100 rerun is authorized merely because source hardening is green.
