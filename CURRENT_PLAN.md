# RealSaS-OPT — Current Plan

**Date:** 2026-10-18  
**Status:** ACTIVE POST-V8 QUALITY / TRUTH / CONSUMPTION PROGRAM  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Review vehicle:** draft PR #116, to be fast-forwarded to this source line  
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

---

## 1. Completed architectural baseline: V8

V8 is complete at step12000 and is the current architectural/measurement baseline, not a scientific PASS.

Supported result:

- sealing the shared carrier after M1 preserved geometry/extraction exactly across downstream specialist training;
- sealed H still supported substantial appearance/relation/topology learning;
- therefore generic semantic classes or generic adapters are not currently justified by capacity evidence.

Do not rerun or reinterpret V8 merely to produce a prettier final number. Preserve it as the baseline against which focused post-V8 experiments are compared.

---

## 2. Workstream A — carrier hardening and CI

### A1. Exact stage identity

**Status: IMPLEMENTED**

The quality contract uses canonical V8 stage IDs and fails on unknown A/R/T/J aliases.

### A2. Runtime optimizer guard

**Status: IMPLEMENTED**

Future V8-line execution validates both configured `scene_lr_scale` and actual `scene_core` optimizer-group LR after every stage application.

### A3. Branch CI

**Status: IMPLEMENTED / GREEN REQUIRED**

The focused vNext workflow runs on relevant pushes to the active post-V8 branch and gates post-V8 quality, boundary, relation-truth-v2 and trainer contracts plus existing native regressions.

No commit is promoted merely because source exists; GitHub Actions must be green.

### A4. Review-line consolidation

**Status: PENDING FINAL FAST-FORWARD**

The historical vNext draft PR #116 is a direct ancestor of the post-V8 branch. Fast-forward the PR head to the active source head after this tranche so repo navigation has one visible vNext review vehicle.

---

## 3. Workstream B — truth cleanup

### B1. Boundary Causality Teacher V1

**Status: CONTRACT IMPLEMENTED / PRODUCER WIRING PENDING**

Allowed causal labels:

```text
GEOMETRIC_TERMINATION
MATERIAL_BOUNDARY
ALPHA_BOUNDARY
OCCLUSION_VISIBILITY_BOUNDARY
SOURCE_PRIMITIVE_COMPONENT_SEAM
RESIDUAL_UNKNOWN
```

`RESIDUAL_UNKNOWN` is abstention and emits no training target.

### B2. Historical relation-truth audit

**Status: AUDITED / CORRECTED**

Two V1-era producer paths are distinct:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

active Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

Therefore V8 cut metrics are about an **owner-or-material boundary proxy**, not pure `1 - same_drawable`, and not authoritative drawable-cut truth.

Audit: `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`.

### B3. Graphics Truth V2

**Status: CONTRACT + LOSS + TESTS IMPLEMENTED / PRODUCER WIRING PENDING**

New files:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

Every scalar relation target now has an explicit future contract for:

- value;
- authority;
- provenance;
- `valid_for_training`.

The V2 loss requires one validity mask per family/sample. Missing truth abstains and cannot silently become a negative target.

Historical V8 shards stay immutable; V2 is a replacement path for new focused work, not a reinterpretation of old bytes.

### B4. Salvage measurable foundation producers

**Status: NEXT IMPLEMENTATION TRANCHE**

Port from `agent/unified-iris-stdgenpp-foundation` only source-defensible producers:

- source geometry/topology provenance;
- material provenance;
- rig/deformation truth;
- all-hit rays;
- fixed-frame dynamic visibility/order;
- residual-boundary diagnostics;
- local geometric continuation evidence.

Do not import owner/amodal/presentation labels that the source cannot identify.

Exit criterion: every new truth row carries authority/provenance and every family reports support/class-balance/abstention statistics.

---

## 4. Workstream C — Compiler/runtime consumption

Machine-readable starting audit:

`docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`

### Already source-proven

- same/cut/boundary influence Compiler fragment clustering;
- visibility becomes runtime visible fragments and changes native raster output;
- front+occlusion become explicit directed order edges;
- native order edges override contradictory physical depth;
- Graphics content, not only a hash, is serialized into `graphics.bin`.

### Open consumption debt

- attachment records are serialized but current native reference raster does not behaviorally use them;
- support-band values are serialized/validated but not behaviorally used by the current native renderer.

Required actions:

1. identify the intended attachment owner in playback/motion runtime;
2. add a behavioral test if it exists, otherwise explicitly mark attachment as transport-only/non-behavioral for this runtime version;
3. either implement an authorized bounded 2D support consumer or remove any claim that support-band data currently affects output;
4. add a fragment-clustering behavioral court showing changed same/cut/boundary evidence changes compiled/runtime fragmentation as expected.

Exit criterion: every learned Graphics signal is classified as `BEHAVIORALLY_USED`, `TRANSPORT_ONLY`, or `UNCONSUMED`.

---

## 5. Workstream D — independent TESSA topology court

**Status: NOT YET BUILT**

Training loss is not qualification.

Build a held-out court over actual proposed topology/surface behavior:

- valid/degenerate face counts;
- connected-component error;
- boundary-loop and hole behavior;
- non-manifold diagnostics without assuming all sources are closed;
- canonical surface deviation;
- source-support coverage;
- stability under held-out views/contexts;
- subject-level and cohort-level worst-case reporting.

Only after this court identifies a concrete failure may we introduce:

- StdGEN++-style sign/hole regularization;
- higher-resolution extraction;
- topology-specific specialist capacity.

Every such change requires ablation and geometry non-regression.

---

## 6. Workstream E — specialist convergence / optimization

V8 proved architectural isolation, not optimal downstream convergence.

For each active head/family record at fixed intervals:

```text
train p50 / p95
held-out p50 / p95
active example count
positive / negative balance
trivial-negative fraction
gradient L2 norm
parameter update L2 norm
update / parameter ratio
optimizer group + LR
best held-out checkpoint
prior-family non-regression metrics
```

Rules:

- separate optimizer groups by evidence family where convergence differs materially;
- do not use one blanket evidence LR as an article of faith;
- diagnose missing gradient, tiny update, class imbalance, trivial negatives and target ambiguity before changing LR;
- allow preregistered stage extension only while held-out improvement remains meaningful;
- select best qualified checkpoint, not automatically the final step;
- keep `scene_core` sealed throughout specialist optimization.

`0.1–0.2` comparable normalized residuals remain optimization debt by default. `~0.05` is a planning reference only, not a universal scientific gate.

---

## 7. Workstream F — StdGEN++ transfer program

Re-audit source:

`docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`

### Adopt now as design discipline

- stable shared carrier / constrained specialization;
- preserve relevant geometric/raster supervision through mesh refinement;
- treat decomposition truth as a data-identifiability problem;
- explicit thin-structure/topology courts;
- specialist branches only where scale/task demands them.

### Investigate after current courts

- sparse coarse-to-fine high-resolution extraction;
- continuity-conditioned hole/sign regularization;
- evidence-specific LoRA/adapters;
- local high-resolution geometry specialist;
- richer TESSA boundary/component summaries.

### Do not copy

- fixed body/hair/cloth inference ontology;
- silent forced SDF-sign repair as authority;
- anatomical branch names as RealSaS architecture;
- hidden/amodal truth that current source assets cannot identify.

Exact StdGEN++ training source is not publicly available in the StdGEN repository as of this state. Paper claims and public-code facts remain separately labeled.

---

## 8. Workstream G — capacity decision court

Do not expand model capacity merely because a residual is high.

A specialist adapter/branch becomes eligible only if:

1. target authority is defensible;
2. held-out sampling and class balance are adequate;
3. gradients and updates are healthy;
4. bounded specialist LR/schedule search does not solve the plateau;
5. sealed-H probes show the required information is not reasonably decodable;
6. earlier capabilities remain non-regressed.

If the bottleneck is head capacity, change the specialist first. Shared-carrier reopening remains the highest-cost option and needs a separate falsification court.

---

## 9. Future focused training

There is no authorization for another blind 12k curriculum run.

Future experiments should be narrow and preregistered, for example:

- relation truth V2 producer + boundary-head optimization;
- relation sampling/class-balance repair;
- topology court-driven regularization;
- coarse-to-fine extraction ablation;
- specialist adapter ablation if a capacity court becomes eligible.

Each experiment must state before running:

- exact hypothesis;
- source authority;
- target family;
- trainable parameter set;
- frozen parameter set;
- held-out court;
- pass/fail gate;
- non-regression gates;
- exact parent checkpoint;
- result that would falsify the intervention.

---

## 10. Execution order

```text
1. carrier policy / canonical stage IDs / runtime optimizer guard       [DONE]
2. focused post-V8 branch CI                                            [DONE; GREEN REQUIRED]
3. StdGEN++ / StdGEN / InstantMesh re-audit                             [DONE; dossier maintained]
4. decision/falsification ledger                                        [DONE; maintain continuously]
5. Graphics source-consumption lineage                                  [DONE initial audit]
6. root navigation/state/agent docs                                     [DONE; maintain]
7. relation producer semantics correction + truth V2 contract           [DONE]
8. consolidate draft PR #116 onto current source head                    [NEXT]
9. salvage identifiable foundation evidence producers                   [NEXT]
10. wire relation truth V2 producer + corpus statistics                  [NEXT]
11. complete attachment/support behavioral ownership audit               [NEXT]
12. independent TESSA topology court                                    [NEXT]
13. convergence telemetry + best-held-out checkpoint machinery          [NEXT]
14. preregister metric-specific gates                                   [AFTER 9–13]
15. focused optimization experiments                                    [AFTER COURTS]
16. capacity/extraction changes only from court evidence                 [CONDITIONAL]
```

No merge to `main` until CI is green, PR state matches these files, and the decision ledger has no unresolved contradiction with code behavior.
