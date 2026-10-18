# RealSaS-OPT — Current Plan

**Date:** 2026-10-18  
**Status:** ACTIVE POST-V8 QUALITY / TRUTH / CONSUMPTION PROGRAM  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Verified pre-docs code head:** `b38feee7f8fafa2308b8e3c9cecdc22f248e28a7` (`39126288809` green)  
**Review vehicle:** draft PR #116; stale until synchronized to the final green active head  
**Production authority:** `main` until reviewed merge  
**Training policy:** no new broad Fit8 run until the courts below justify it

## 0. Non-negotiable architecture rules

### Single canonical carrier

```text
K x candidate M*
        ↓
static admission + candidate mechanics
        ↓
deterministic selection
        ↓
ONE canonical M/G/W_M
        ↓
Graphics + Motion + Runtime
```

Candidates are evidence only. A second deformable visual mesh remains forbidden.

### Post-M1 shared-carrier seal

```text
Geometry + Mechanics -> shared H -> M1 seal -> A/R/T/J specialist learning
```

A1/R1/T1/J1 may not optimize `scene_core` unless a separately preregistered carrier-falsification court authorizes reopening and specifies non-regression gates.

V8 already established this structural seal. Historical V8 physically grouped mechanical/appearance/relation parameters under one `evidence_heads` AdamW group, while stage objectives isolated which head received gradients. Post-V8 `optimization_v2.py` adds separate physical groups/LRs/gradient-update telemetry for mechanical, appearance, relation and structure; it is diagnostic/optimization hardening, not a new carrier-freeze idea.

### Compiler / truth / conditioning laws

> Compiler may globally reconcile admissible evidence; it may not hide model incapacity or fabricate unavailable truth.

> Unknown is a first-class outcome. Unidentifiable owner/amodal/presentation targets must abstain.

> View-conditioned targets require matching prediction conditioning unless a court proves view invariance.

---

## 1. Completed baseline: V8

V8 completed step12000 and remains the architectural/measurement baseline, not a scientific PASS.

Supported result: post-M1 carrier sealing preserved geometry/extraction while appearance/relation/topology objectives remained learnable from sealed H. This weakens any present claim that generic semantic classes, LoRA, or broad carrier reopening are required.

---

## 2. Workstream A — source hardening / CI

- canonical post-M1 stage IDs + fail-closed scene guard: **DONE**;
- TESSA default-offset regression: **FIXED / CI-GATED**;
- focused vNext workflow: **GREEN at `b38feee7...`, run `39126288809`**;
- specialist optimizer V2: **IMPLEMENTED / GREEN** — `scene_core=0`, separate mechanical/appearance/relation/structure groups, explicit LRs, per-group telemetry;
- review-line consolidation: **PERSISTENT GATE** — PR #116 is stale until synchronized to a final green head.

### Self-hosted stale-run quarantine

Run `39121291171` / `iris-vnext-fit8-prepare-self-hosted-v1 #22` executed stale head `90c9c2f...` when `realsas-wsl-1660ti` came online. It produced a valid-for-that-source local preparation and then started copying to `gdrive:RealSaS_Fit8_IRIS_vNext_20261016`. Cancellation occurred during Drive persistence before round-trip SHA verification.

Until explicit remote verification:

- treat the Drive preparation path as **QUARANTINED / UNVERIFIED**;
- do not consume it for current experiments;
- keep the self-hosted runner offline;
- preserve GitHub artifact `12137271848` only as forensic evidence of the stale run;
- build current-branch verification/preparation ownership before another self-hosted execution.

---

## 3. Workstream B — truth + conditioning cleanup

### Boundary Causality V1

Contract implemented. `RESIDUAL_UNKNOWN` abstains. Producer integration remains a current data task.

### Relation Truth V2 — source stack implemented

Implemented and CI-green source:

- `models/iris/vnext/graphics_truth_v2.py`;
- `models/iris/vnext/training_v2.py`;
- `tools/iris_vnext/build_fit8_relation_truth_v2.py`;
- `models/iris/vnext/relation_support_v2.py`;
- `models/iris/vnext/heads_v2.py` + deterministic V1→V2 migration semantics;
- `models/iris/vnext/optimization_v2.py`.

`GraphicsRelationHeadV2` keeps static same/cut/attachment separate and moves projected alpha/appearance boundaries to the camera-conditioned view branch with front/occlusion/visibility. New projected-boundary output rows start from neutral logits while compatible V1 rows are preserved.

**Not yet done:** an actual Fit8 Truth V2 corpus has not been shown materialized/sealed in repository authority; a real V8 checkpoint has not yet been migrated/sealed as a V2 checkpoint receipt; focused optimization has not run; no scientific PASS exists.

Truth V2 static same/cut/attachment abstain without independent authority. Current valid occlusion rows have single-class positive support, so they are not eligible for a binary held-out PASS court. Preserve that fact; do not synthesize negative truth.

### Foundation salvage — exact source map

Provenance branch: `agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`.

Eligible for selective reuse after current-branch review:

- geometry/topology: `tools/corpus/qualify_f_geometry_carrier_v1.py`, `fit8_surface_topology_invariants_v1.py`;
- glTF render/geometric equivalence + material/skin provenance: `fit8_render_vertex_equivalence_v1.py`, `fit8_gltf_static_carrier_v1.py`;
- FBX exact-source census: `ufbx_static_census_probe_v2.c`, `ufbx_static_census_probe_v3.c`, `fit8_fbx_static_carrier_v1.py`, `fit8_fbx_static_carrier_v2.py`;
- all-hit static rays: `fit8_static_ray_census_v1.py`, `fit8_static_ray_census_v2.py`, `fit8_static_ray_census_v3.py`;
- authored-motion fixed-frame evidence: `fit8_gltf_dynamic_pose_v1.py`, `fit8_dynamic_fixed_frame_presentation_v1.py`, `fit8_dynamic_mechanics_metrics_v1.py`, `fit8_gltf_dynamic_census_v2.py`;
- residual boundary: `fit8_f_residual_boundary_graph_v1.py`, `fit8_f_residual_signature_v1.py`, `fit8_face_strand_boundary_incidence_v1.py`;
- local continuation: `fit8_local_patch_teacher_v1.py`, `fit8_patch_face_support_v1.py`, `fit8_f_patch_continuation_court_v1.py`, `fit8_f_face_qualified_continuation_court_v1.py`.

Do **not** port owner/amodal/presentation/semantic-partition claims as truth merely because related foundation code exists. Generic FIT8 3D does not uniquely identify intended drawable ownership, hidden support, intended 2D presentation override or global named ontology.

Exit criterion for new truth: every family reports authority/provenance, valid support, positive/negative balance and abstention statistics.

---

## 4. Workstream C — Compiler/runtime consumption

Audit: `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Already source-proven: fragment clustering from same/cut/boundary, visibility changing final native RGBA, front+occlusion producing directed order, native order overriding contradictory physical depth, and Graphics content serialized into `graphics.bin`.

Open debt: attachment remains transport-only for current native raster behavior; support-band values are serialized/validated but not applied by the reference renderer. Close every family through `DEFINED → PRODUCED → SERIALIZED → CONSUMED → BEHAVIORALLY_USED`, or explicitly label the missing stage.

---

## 5. Workstream D — independent TESSA topology court

**NOT YET SEALED.** Training loss is not qualification.

Court must report actual valid/degenerate faces, component error, boundary loops/holes, non-manifold diagnostics where meaningful, canonical-surface deviation, source support, held-out stability and worst-case subject/cohort behavior.

Only after a concrete failure is measured may StdGEN++-style sign/hole regularization, sparse high-resolution extraction or topology-specific capacity become eligible; require ablation + geometry non-regression.

---

## 6. Workstream E — specialist convergence

For each authoritative valid family record train/held-out p50+p95, active count, class balance, abstention, gradient L2, parameter/update L2, update/parameter ratio, optimizer group/LR, best held-out checkpoint and prior-family non-regression.

Ordering rule:

1. authority;
2. conditioning;
3. sampling/class support;
4. gradients/updates;
5. LR/schedule;
6. capacity only after the above.

`scene_core` remains sealed. `0.1–0.2` comparable residuals remain optimization debt by default; `~0.05` is only a historical planning reference for genuinely comparable normalized metrics.

---

## 7. StdGEN++ transfer program

Pinned public-code references:

- StdGEN `f1b0c822689dc2539c774bb73090a67c66e2e766`;
- InstantMesh `08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.

Adopt discipline now: stable carrier, constrained specialization, retained supervision, explicit topology courts, and special data for hidden/internal truth. Investigate coarse-to-fine extraction or continuity regularization only after the relevant court. Generic LoRA/adapters are not authorized before a clean-target capacity court fails. Do not copy the body/hair/cloth ontology.

The public StdGEN repo lacks the exact full training loop; paper-only training/curriculum claims remain `PAPER_EXACT`, not `CODE_EXACT`.

---

## 8. Immediate execution order

```text
1. docs/state/ledger/source-map synchronization                         [THIS COMMIT]
2. verify/quarantine stale Drive preparation from run 39121291171
3. make current branch own safe self-hosted verification/preparation flow
4. materialize Relation Truth V2 sidecars/manifest on exact admitted Fit8 inputs
5. audit per-family validity/support/class balance/abstention
6. seal real V8 relation-head -> GraphicsRelationHeadV2 migration receipt/checkpoint
7. preregister focused frozen-carrier relation/boundary optimization
8. run focused optimization with head-specific telemetry
9. build independent TESSA topology court
10. close Compiler/runtime behavioral-consumption gaps
11. synchronize PR #116 to the final green review head
12. capacity/extraction/LoRA changes only from court evidence
```

No blind 12k curriculum run and no broad A100 notebook is authorized by source hardening alone. No merge to `main` until the final review head is green and the decision ledger matches code behavior.
