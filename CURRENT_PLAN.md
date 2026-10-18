# RealSaS-OPT — Current Plan

**Date:** 2026-10-18  
**Status:** ACTIVE POST-V8 QUALITY / TRUTH / CONSUMPTION PROGRAM  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Review vehicle:** draft PR #116; its head must match the active source head before review/merge  
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
Geometry + Mechanics
        ↓
shared carrier H
        ↓
M1 seal
        ↓
Appearance / Relations / TESSA / Joint specialist learning
```

A1/R1/T1/J1 may not optimize `scene_core` unless a separately preregistered carrier-falsification court explicitly authorizes reopening it and specifies non-regression gates.

### Compiler law

> Compiler may enforce and globally reconcile admissible evidence; Compiler may not hide model incapacity or fabricate unavailable truth.

### Truth law

> Unknown is a first-class outcome. An unidentifiable owner/amodal/presentation target must abstain instead of inheriting a convenient geometric proxy.

### Conditioning law

> A target derived from view-conditioned evidence must not be assigned to a camera-free prediction path unless a court proves that target is view-invariant under the chosen representation.

---

## 1. Completed architectural baseline: V8

V8 is complete at step12000 and is the current architectural/measurement baseline, not a scientific PASS.

Supported result:

- sealing the shared carrier after M1 preserved geometry/extraction exactly across downstream specialist training;
- sealed H still supported substantial appearance/relation/topology learning;
- generic semantic classes or generic adapters are not currently justified by capacity evidence.

Do not rerun or reinterpret V8 merely to produce a prettier final number. Preserve it as the baseline against which focused post-V8 experiments are compared.

---

## 2. Workstream A — carrier hardening and CI

### A1. Exact stage identity — IMPLEMENTED

The quality contract uses canonical V8 stage IDs and fails on unknown A/R/T/J aliases.

### A2. Runtime optimizer guard — IMPLEMENTED

Future V8-line execution validates both configured `scene_lr_scale` and actual `scene_core` optimizer-group LR after every stage application.

### A3. TESSA default-offset regression — FIXED / CI REQUIRED

Focused CI found that `structure_v2.py` defaulted `sequence_position_offset` to `None` while the decoder contract expects zero or an explicit offset. Default/explicit-`None` now normalize to `0`; the existing direct topology contract remains in CI.

### A4. Branch CI — IMPLEMENTED / GREEN REQUIRED

The focused vNext workflow runs on relevant pushes to the active branch and gates post-V8 quality, boundary, relation-truth-v2, trainer and existing native regressions.

### A5. Review-line consolidation — PERSISTENT GATE

Draft PR #116 is the canonical vNext review vehicle. Before review or merge, verify its head exactly equals the final green active source head. If the active source advances after synchronization, the PR is out of date by definition and must be fast-forwarded again. This is a gate, not a one-time milestone.

---

## 3. Workstream B — truth + conditioning cleanup

### B1. Boundary Causality Teacher V1 — CONTRACT IMPLEMENTED / PRODUCER PENDING

Allowed causes:

```text
GEOMETRIC_TERMINATION
MATERIAL_BOUNDARY
ALPHA_BOUNDARY
OCCLUSION_VISIBILITY_BOUNDARY
SOURCE_PRIMITIVE_COMPONENT_SEAM
RESIDUAL_UNKNOWN
```

`RESIDUAL_UNKNOWN` is abstention and emits no target.

### B2. Historical relation producer audit — AUDITED / CORRECTED

Two historical paths are distinct:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

V8 cut metrics therefore measure an **owner-or-material boundary proxy**, not pure `1 - same_drawable`, and not authoritative drawable-cut truth.

### B3. Relation-boundary conditioning mismatch — CONFIRMED / MUST REPLACE BEFORE LR SEARCH

V8's shard builder derives `relation_alpha_boundary_probability` and part of `relation_appearance_boundary_probability` from the current render view's local 3×3 raster alpha/RGB neighborhood.

But `GraphicsRelationHeadV1.static_relations(feature_a, feature_b)` predicts those boundary probabilities without camera input. The trainer compares these camera-free outputs against view-derived targets.

Therefore projected relation-boundary residuals combine:

- truth/causality debt;
- a view-conditioning contract mismatch;
- only then optimization/capacity debt.

Do not tune relation-boundary LR/capacity against this V1 contract.

Replacement direction:

```text
static structural pair branch
    qualified ownership/same when identifiable
    structural cut when identifiable
    attachment/contact
    material/geometric structural boundary causes

view-conditioned pair branch
    front
    occlusion
    visibility
    projected alpha boundary
    projected appearance boundary
```

`AppearanceEvidenceHeadV1` is already camera-conditioned and remains a separate per-point path; its boundary debt must be analyzed independently.

### B4. Graphics Truth V2 — CONTRACT + LOSS + TESTS IMPLEMENTED / PRODUCER PENDING

New files:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

Every scalar relation target carries value, authority, provenance and `valid_for_training`. The V2 loss requires explicit per-family/sample validity masks; missing truth abstains rather than becoming a negative target.

Historical V8 shards/head stay immutable. New relation training requires a versioned producer and versioned view-conditioned relation path.

### B5. Salvage measurable foundation producers — NEXT IMPLEMENTATION TRANCHE

Port from `agent/unified-iris-stdgenpp-foundation` only source-defensible producers:

- source geometry/topology provenance;
- material provenance;
- rig/deformation truth;
- all-hit rays;
- fixed-frame dynamic visibility/order;
- residual-boundary diagnostics;
- local geometric continuation evidence.

Do not import owner/amodal/presentation labels that the source cannot identify.

Exit criterion: every new truth row carries authority/provenance and every family reports support, positive/negative balance and abstention statistics.

---

## 4. Workstream C — Compiler/runtime consumption

Audit: `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Already source-proven:

- same/cut/boundary influence fragment clustering;
- visibility becomes runtime visible fragments and changes native raster output;
- front+occlusion become directed order edges;
- native order edges override contradictory physical depth;
- Graphics content, not only a hash, is serialized into `graphics.bin`.

Open consumption debt:

- attachment is serialized but current native reference raster does not behaviorally use it;
- support-band values are serialized/validated but not behaviorally used by the current native renderer.

Required actions:

1. identify intended attachment owner in playback/motion runtime;
2. add behavioral test if it exists, otherwise mark attachment transport-only for this runtime version;
3. wire an explicitly authorized bounded 2D support consumer or remove behavioral claims for support-band;
4. add a fragment-clustering behavioral court showing changed same/cut/boundary evidence changes compiled/runtime fragmentation.

---

## 5. Workstream D — independent TESSA topology court

**Status: NOT YET BUILT**

Training loss is not qualification.

Build a held-out court over actual topology/surface behavior:

- valid/degenerate face counts;
- connected-component error;
- boundary-loop and hole behavior;
- non-manifold diagnostics without assuming all sources are closed;
- canonical-surface deviation;
- source-support coverage;
- stability under held-out views/contexts;
- subject/cohort worst-case reporting.

Only after a concrete failure is measured may we introduce StdGEN++-style sign/hole regularization, higher-resolution extraction or topology-specific capacity. Every intervention needs ablation + geometry non-regression.

---

## 6. Workstream E — specialist convergence / optimization

V8 proved architectural isolation, not optimal downstream convergence.

Required telemetry per valid target family:

```text
train p50 / p95
held-out p50 / p95
active example count
positive / negative balance
abstention rate
trivial-negative fraction
gradient L2 norm
parameter update L2 norm
update / parameter ratio
optimizer group + LR
best held-out checkpoint
prior-family non-regression metrics
```

Ordering rule:

1. verify target authority;
2. verify conditioning contract;
3. verify sampling/class balance;
4. verify gradients/updates;
5. only then tune LR/schedule/capacity.

Keep `scene_core` sealed throughout specialist optimization.

`0.1–0.2` comparable normalized residuals remain optimization debt by default. `~0.05` is a planning reference only, not a universal scientific gate.

---

## 7. Workstream F — StdGEN++ transfer program

Re-audit: `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`.

Adopt now as discipline:

- stable shared carrier / constrained specialization;
- preserve relevant geometric/raster supervision through mesh refinement;
- treat decomposition truth as a data-identifiability problem;
- explicit thin-structure/topology courts;
- specialist branches only where scale/task demands them.

Investigate after current courts:

- sparse coarse-to-fine high-resolution extraction;
- continuity-conditioned hole/sign regularization;
- evidence-specific LoRA/adapters;
- local high-resolution geometry specialist;
- richer TESSA boundary/component summaries.

Do not copy fixed body/hair/cloth ontology, forced-SDF-sign repair as authority, or hidden/amodal truth unavailable from source assets.

---

## 8. Workstream G — capacity decision court

Do not expand model capacity merely because a residual is high.

A specialist adapter/branch becomes eligible only if:

1. target authority is defensible;
2. conditioning matches target semantics;
3. held-out sampling/class balance is adequate;
4. gradients/updates are healthy;
5. bounded specialist LR/schedule search fails;
6. sealed-H probes show the signal is not reasonably decodable;
7. earlier capabilities remain non-regressed.

Modify the specialist first. Shared-carrier reopening remains the highest-cost option.

---

## 9. Future focused training

There is no authorization for another blind 12k curriculum run.

Future experiments must be narrow and preregistered and must state:

- exact hypothesis;
- source authority;
- target family and conditioning semantics;
- trainable/frozen parameter sets;
- held-out court;
- pass/fail gate;
- non-regression gates;
- exact parent checkpoint;
- result that would falsify the intervention.

---

## 10. Execution order

```text
1. carrier policy / canonical stage IDs / runtime optimizer guard        [DONE]
2. focused branch CI + TESSA default-offset regression fix              [GREEN REQUIRED]
3. StdGEN++ / StdGEN / InstantMesh re-audit                              [DONE; maintain]
4. decision/falsification ledger                                         [DONE; maintain]
5. Graphics source-consumption lineage                                   [DONE initial audit]
6. root navigation/state/agent docs                                      [DONE; maintain]
7. relation producer semantics + Graphics Truth V2                       [DONE contract]
8. relation boundary conditioning mismatch audit                          [DONE]
9. verify/sync draft PR #116 to the final green source head              [REVIEW GATE]
10. salvage identifiable foundation evidence producers                   [NEXT]
11. wire Truth V2 producer + view-conditioned relation schema + stats    [NEXT]
12. close attachment/support behavioral ownership audit                   [NEXT]
13. independent TESSA topology court                                     [NEXT]
14. convergence telemetry + best-held-out checkpoint machinery           [NEXT]
15. preregister metric-specific gates                                    [AFTER 10–14]
16. focused optimization experiments                                     [AFTER COURTS]
17. capacity/extraction changes only from court evidence                  [CONDITIONAL]
```

No merge to `main` until the final review head is green and the decision ledger has no unresolved contradiction with code behavior.
