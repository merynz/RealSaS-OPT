# RealSaS-OPT — Current Plan

**Date:** 2026-10-18  
**Status:** ACTIVE POST-V8 QUALITY / TRUTH / CONSUMPTION PROGRAM  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Latest verified code parent before this docs sync:** `b6d77d1805d17ffb59e19d6a8b8626d8d8e6cb69` (`39130204378` green)  
**Review vehicle:** draft PR #116; fast-forward only after this docs head is green  
**Production authority:** `main` until reviewed merge  
**Training policy:** no new broad Fit8 run until focused courts justify it

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

### Post-M1 carrier seal

```text
Geometry + Mechanics -> shared H -> M1 seal -> A/R/T/J specialist learning
```

A1/R1/T1/J1 may not optimize `scene_core` unless a separately preregistered carrier-falsification court authorizes reopening and defines non-regression gates.

V8 already established this structural seal. Historical V8 physically grouped mechanical/appearance/relation parameters under one `evidence_heads` AdamW group while stage objectives isolated which specialist received gradients. Post-V8 `optimization_v2.py` adds separate physical groups/LRs/gradient-update telemetry; it is diagnostic/optimization hardening, not a new freeze idea.

### Compiler / truth / conditioning laws

> Compiler may globally reconcile admissible evidence; it may not hide model incapacity or fabricate unavailable truth.

> Unknown is first-class. Unidentifiable owner/amodal/presentation targets abstain.

> View-conditioned targets require matching prediction conditioning unless a court proves view invariance.

---

## 1. Completed baseline: V8

V8 completed step12000 and remains the architecture/measurement baseline, not a scientific PASS.

Supported result: post-M1 carrier sealing preserved geometry/extraction while appearance/relation/topology objectives remained learnable from sealed H. This weakens any present claim that generic semantic classes, generic LoRA, or broad carrier reopening are required.

---

## 2. Workstream A — source hardening / CI

Current status: **GREEN at `b6d77d...`, run `39130204378`.**

Completed source hardening:

- canonical post-M1 stage IDs + fail-closed scene guard;
- TESSA default-offset regression repair + gate;
- specialist optimizer V2 with `scene_core=0`, separate mechanical/appearance/relation/structure groups, explicit LRs, per-group telemetry;
- Graphics Truth V2 contract/loss/producer source;
- Relation Head V2 conditioning/migration source;
- support/class-balance court eligibility source;
- native Graphics regression coverage;
- self-hosted stale-writer retirement + read-only quarantine audit contract.

Review-line consolidation remains a gate: PR #116 must be fast-forwarded only to the **final green docs/source head**.

---

## 3. Workstream B — self-hosted / remote preparation quarantine

Incident run: `39121291171` / prepare #22 / stale source `90c9c2f...`.

Affected remote:

`gdrive:RealSaS_Fit8_IRIS_vNext_20261016`

The run was cancelled during remote persistence before round-trip verification. Therefore the remote is **QUARANTINED / UNVERIFIED**.

Safety source now green:

- legacy `iris_vnext_fit8_prepare_self_hosted_v1.yml` is retired and cannot queue a self-hosted writer from push;
- it contains no `rclone` write path;
- `iris_vnext_fit8_quarantine_audit_self_hosted_v2.yml` is manual-dispatch only, active-branch/exact-SHA bound and read-only;
- `audit_fit8_preparation_snapshot_v1.py` records all downloaded file SHA256 values and source binding;
- read-only audit **never authorizes consumption**, even if remote source SHA matches current source;
- forensic output is uploaded as a GitHub artifact.

Immediate rule: keep runner offline until we deliberately dispatch the read-only audit. There is currently **no authorized self-hosted preparation writer**.

After the read-only audit:

1. classify actual remote contents;
2. preserve forensic evidence;
3. build a new manual SHA-scoped candidate preparation writer;
4. write to a commit-SHA-specific candidate remote, not the quarantined shared path;
5. require full local↔remote round-trip hash equality before the candidate can be considered for later authority binding;
6. do not auto-promote to a canonical shared remote.

---

## 4. Workstream C — truth + conditioning cleanup

### Boundary Causality V1

Contract implemented. `RESIDUAL_UNKNOWN` abstains. Populated corpus remains pending.

### Relation Truth V2

Implemented and CI-green source:

- `models/iris/vnext/graphics_truth_v2.py`;
- `models/iris/vnext/training_v2.py`;
- `tools/iris_vnext/build_fit8_relation_truth_v2.py`;
- `models/iris/vnext/relation_support_v2.py`;
- `models/iris/vnext/heads_v2.py`;
- `models/iris/vnext/optimization_v2.py`.

`GraphicsRelationHeadV2` keeps static same/cut/attachment separate and moves projected alpha/appearance boundaries to the camera-conditioned view branch with front/occlusion/visibility. Compatible V1 rows migrate; new projected-boundary rows start neutral.

Still pending:

- actual Fit8 Truth V2 corpus materialization/seal;
- real V8 checkpoint→V2 migration receipt/checkpoint;
- focused optimization;
- held-out scientific PASS.

Static same/cut/attachment abstain without independent authority. Current valid occlusion rows are single-class positive support and therefore binary-court ineligible. Do not synthesize negative truth.

### Foundation salvage — exact source map

Provenance: `agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`.

Selective source-defensible paths:

```text
geometry/topology
  tools/corpus/qualify_f_geometry_carrier_v1.py
  tools/corpus/fit8_surface_topology_invariants_v1.py

render/geometric + material/skin provenance
  tools/corpus/fit8_render_vertex_equivalence_v1.py
  tools/corpus/fit8_gltf_static_carrier_v1.py
  tools/corpus/ufbx_static_census_probe_v2.c
  tools/corpus/ufbx_static_census_probe_v3.c
  tools/corpus/fit8_fbx_static_carrier_v1.py
  tools/corpus/fit8_fbx_static_carrier_v2.py

static all-hit evidence
  tools/corpus/fit8_static_ray_census_v1.py
  tools/corpus/fit8_static_ray_census_v2.py
  tools/corpus/fit8_static_ray_census_v3.py

authored-motion fixed-frame evidence
  tools/corpus/fit8_gltf_dynamic_pose_v1.py
  tools/corpus/fit8_dynamic_fixed_frame_presentation_v1.py
  tools/corpus/fit8_dynamic_mechanics_metrics_v1.py
  tools/corpus/fit8_gltf_dynamic_census_v2.py

residual boundary
  tools/corpus/fit8_f_residual_boundary_graph_v1.py
  tools/corpus/fit8_f_residual_signature_v1.py
  tools/corpus/fit8_face_strand_boundary_incidence_v1.py

local continuation
  tools/corpus/fit8_local_patch_teacher_v1.py
  tools/corpus/fit8_patch_face_support_v1.py
  tools/corpus/fit8_f_patch_continuation_court_v1.py
  tools/corpus/fit8_f_face_qualified_continuation_court_v1.py
```

Do not port owner/amodal/presentation/semantic-partition claims as generic FIT8 truth.

Exit criterion for new truth: authority/provenance, valid support, positive/negative balance and abstention statistics per family.

---

## 5. Workstream D — Compiler/runtime consumption

Audit: `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Already source-proven:

- fragment clustering from same/cut/boundary;
- visibility changing final native RGBA;
- front+occlusion producing directed order;
- native order overriding contradictory physical depth;
- Graphics content serialized into `graphics.bin`.

Open debt:

- attachment remains transport-only for current native raster behavior;
- support-band values are serialized/validated but not applied by the reference renderer.

Every family must end with an explicit `BEHAVIORALLY_USED`, `TRANSPORT_ONLY`, or `UNCONSUMED` classification.

---

## 6. Workstream E — independent TESSA topology court

**NOT YET SEALED.** Training loss is not qualification.

Court must report actual valid/degenerate faces, component error, boundary loops/holes, meaningful manifold diagnostics, canonical-surface deviation, source support, held-out stability and worst-case cohort behavior.

Only after a concrete failure is measured may StdGEN++-style sign/hole regularization, sparse high-resolution extraction or topology-specific capacity become eligible; require ablation + geometry non-regression.

---

## 7. Workstream F — specialist convergence

For each authoritative valid family record:

- train and held-out p50/p95;
- active count, class balance, abstention/trivial-negative fraction;
- gradient L2;
- parameter/update L2 and update/parameter ratio;
- optimizer group/LR;
- best held-out checkpoint;
- prior-family non-regression.

Diagnosis order:

```text
authority
→ conditioning
→ sampling/class support
→ gradients/updates
→ LR/schedule
→ capacity last
```

`scene_core` remains sealed. `0.1–0.2` comparable residuals remain optimization debt by default; historical `~0.05` is only planning context for genuinely comparable normalized metrics.

---

## 8. StdGEN++ transfer program

Pinned public code:

- StdGEN `f1b0c822689dc2539c774bb73090a67c66e2e766`;
- InstantMesh `08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.

Adopt now as discipline: stable carrier, constrained specialization, retained supervision, explicit topology courts, special data for hidden/internal truth.

Conditional after courts: coarse-to-fine extraction, continuity/hole regularization, evidence-specific adapters/LoRA, local high-resolution specialist.

Do not copy body/hair/cloth ontology. Public StdGEN lacks the exact full training loop; paper-only claims remain `PAPER_EXACT`.

---

## 9. Immediate execution order

```text
1. commit this state/plan sync and require green CI
2. fast-forward PR #116 branch to the final green head without force
3. confirm no queued self-hosted write jobs
4. deliberately dispatch read-only quarantine audit on the final active head
5. inspect audit artifact and classify remote contamination/staleness
6. design SHA-scoped candidate preparation writer + full round-trip verifier
7. materialize Relation Truth V2 on exact admitted Fit8 inputs
8. audit per-family validity/support/class balance/abstention
9. seal real V8 relation-head -> GraphicsRelationHeadV2 migration receipt/checkpoint
10. preregister and run focused frozen-carrier relation/boundary optimization
11. build independent TESSA topology court
12. close Compiler/runtime behavioral-consumption gaps
13. capacity/extraction/LoRA changes only from court evidence
```

No blind 12k curriculum run and no broad A100 notebook is authorized by source hardening alone. No merge to `main` until review is complete and final source/docs state is coherent.
