# RealSaS-OPT — Current State

**Date:** 2026-10-18  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Review vehicle:** draft PR #116 is the vNext review vehicle and must be fast-forwarded to the active source line before merge  
**Active program:** **POST-V8 QUALITY / TRUTH / CONSUMPTION QUALIFICATION**  
**Scientific status:** **NO IRIS FIT8 SCIENTIFIC PASS**

## 1. Authority separation

Do not collapse these into one notion of “current”:

1. `main` is still the merged production-code authority.
2. The completed V8 Fit8 run is the current architectural/measurement baseline. Its source authority is commit `5083c9915533ec8ec6530c7d229c635887abfea7` plus its exact external checkpoint/receipt lineage. Later source edits do not retroactively mutate that completed artifact.
3. The active post-V8 research branch contains hardening, truth cleanup, audits and future replacement code. It is review evidence until merged.
4. Artifact Registry / Attempt / ProductRevision remain durable experiment/product authority. Git branch chronology is not a substitute.

## 2. What the training lineage established

### V5 — falsified single-objective rewriting of shared H

A1 learned appearance while shared representation quality collapsed. Mechanics degraded severely and the later R1 diagnostic reached an empty FlexiCubes surface.

Conclusion: a later evidence objective may improve its own loss while destroying previously useful representation.

### V6 — cumulative retention was not enough

Cumulative field/extraction/mechanical retention preserved mechanics much better, but A1 still rewrote geometry/extraction. Extraction p95 moved from the roughly `0.20` M1-end regime into the `0.5+` regime and extracted meshes inflated dramatically.

Conclusion: loss balancing is not a sufficient carrier-protection mechanism.

### V8 — supported structural carrier isolation

V8 migrated from the exact V6 M1-end step3600 state and sealed `scene_core` for A1/R1/T1/J1. The run reached global step 12000.

Across step3600→12000:

- geometry p95 remained `0.151504`;
- extraction p95 remained `0.199951`;
- per-subject extraction vertex counts remained unchanged;
- appearance continued improving;
- current relation targets remained strongly decodable;
- topology training objective continued improving.

V8 therefore supports the following current invariant:

```text
G / M establish shared carrier H
        ↓
M1 seal
        ↓
A / R / T / J specialist learning
        ↓
scene_core remains sealed
```

This is **structural isolation**, not temporary loss balancing.

V8 is not a scientific quality PASS because metric-specific preregistered thresholds and an independent topology court were absent.

## 3. Carrier hardening now exists in executable source

`models/iris/vnext/quality_policy_v1.py` now uses the exact V8 stage IDs:

```text
A1_APPEARANCE_EVIDENCE
R1_ROLE_FREE_RELATIONS
T1_MECHANICAL_STRUCTURE
J1_JOINT_FINETUNE
```

Unknown post-M1 aliases fail closed rather than silently bypassing the policy.

`IRISVNextFit8TrainerV8._apply_stage_lrs()` now validates both:

- the stage-level `scene_lr_scale`;
- the actual optimizer `scene_core` group LR.

Any downstream non-zero scene-core LR requires an explicit reopening court. The completed historical V8 artifact remains bound to `5083...`; this source hardening governs future execution only.

## 4. Current IRIS architecture

Active model namespace: `models/iris/vnext/`.

```text
8 x RGBA + cameras
        ↓
frozen visual encoder
        ↓
shared scene carrier
        ↓
       H(x)
   ┌────┼─────────────┐
   ↓    ↓             ↓
Geometry Mechanics  Specialist evidence
field    demand      appearance / relations / structure
   └────┬─────────────┘
        ↓
  Evidence Package
        ↓
     Compiler
   ┌────┼─────────┐
   ↓    ↓         ↓
fragment order  contact/attachment
 graph   graph      graph
   └────┬─────────┘
        ↓
 product/runtime
```

There is no required named body-part taxonomy in the inference contract.

The sole canonical mechanical carrier remains `M/G/W_M`. No independent deformable visual mesh is allowed.

## 5. What frozen H currently tells us

V8 weakened the hypothesis that a named semantic head is required for graphics structure.

Under the current teacher definitions, a sealed H still supports learning of:

- appearance/RGBA evidence;
- same-drawable affinity;
- attachment affinity;
- front/back relation;
- occlusion;
- visibility;
- topology objective.

This proves **decodability under current targets**, not that those targets are all semantically authoritative.

Near-zero relation residual is therefore capacity evidence, not automatically graphics correctness.

## 6. Current highest-priority truth debt: boundary causality

The poor V8 family is not random. `appearance_boundary` remained weak and `cut` remained materially worse than front/visibility/occlusion.

A boundary may arise from:

- geometric termination;
- authored material transition;
- alpha transition;
- occlusion/visibility boundary;
- source primitive/component seam;
- drawable ownership/presentation logic that current generic FIT8 3D may not identify.

`BoundaryCausalityTeacherV1` now admits only provable causes and uses `RESIDUAL_UNKNOWN` as abstention. Unknown is not a negative label and not a training target.

### Newly confirmed debt in the historical relation teacher

`models/iris/vnext/graphics_truth_v1.py` currently constructs adjacent-pair:

```text
same_drawable = (anonymous_drawable_id_a == anonymous_drawable_id_b)
cut           = 1 - same_drawable
```

Therefore the historical `cut` target is a tautological complement of the current owner proxy, not independent causal cut truth.

Decision:

- preserve this schema for V8 provenance;
- do not reinterpret its low/high loss as independent cut correctness;
- replace it with validity-aware/abstaining cut supervision in the next truth schema.

## 7. Foundation branch disposition

Research branch retained for evidence:

`agent/unified-iris-stdgenpp-foundation` @ `d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`

Salvage:

- exact source geometry/topology provenance;
- rig/skin/deformation evidence;
- authored material provenance when present;
- static all-hit ray stacks;
- fixed-frame dynamic visibility/order;
- geometric front/occlusion/visibility;
- component census;
- residual-boundary analysis;
- local geometric continuation as evidence;
- subject-local partition contracts where authority is explicit.

Do not promote from generic FIT8 3D alone:

- authoritative drawable-owner truth;
- hidden/amodal support truth;
- intentional 2D presentation override;
- global body-part semantic identities;
- inferred hidden continuation as ground truth.

The branch falsified the stronger “we can recover full presentation/amodal ownership from generic 3D if we try harder” program. That was an identifiability limit, not merely an implementation bug.

## 8. Compiler/runtime consumption audit

Machine-readable audit:

`docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`

Current source audit:

- `same_drawable` → Compiler fragment clustering → runtime face/fragment ownership → **behavioral path exists**;
- `cut` → fragment-union gate → **behavioral path exists**, but target has truth debt;
- alpha/appearance boundary → fragment-union gate → **behavioral path exists**;
- `front` + `occlusion` → directed order edges → native pixel ordering → **behaviorally used**;
- visibility → visible fragments → native `face_visible()` skip → **behaviorally used and native E2E proven**;
- attachment → Compiler + RuntimeGraphics + `graphics.bin` → **serialized but current native reference raster does not behaviorally use it**;
- support-band pixels → serialized and validated → **not currently behaviorally consumed by the native reference renderer**.

Native ordering contract explicitly proves a graph edge can override contradictory physical depth. Native E2E proves changing Graphics visibility content changes final RGBA output.

Do not claim attachment/support-band behavior until their owning consumer is explicit and tested.

## 9. TESSA/topology state

TESSA topology training loss improved strongly in V8, but there is no independent sealed held-out topology validity court.

Required court must measure, where applicable:

- valid/degenerate faces;
- connected-component error;
- boundary loops / holes;
- non-manifold diagnostics without assuming all sources are closed;
- canonical surface deviation;
- source-support coverage;
- stability across held-out contexts/views.

Training loss is not topology qualification.

## 10. StdGEN++ re-audit state

Detailed dossier:

`docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`

Current conclusions:

- public StdGEN code confirms LoRA-constrained transformer adaptation plus trainable synthesizer/decoders;
- semantic, SDF, RGB, deformation and FlexiCubes outputs share triplane features;
- semantic probabilities actively modify extracted equivalent SDFs, so semantics perform structural functions, not only classification;
- StdGEN++ paper uses dedicated full-body/facial LoRA branches, three supervision stages, multilayer semantic renders, high-resolution coarse-to-fine extraction and a thin-structure hole regularizer;
- exact StdGEN/StdGEN++ training implementation is not public in the StdGEN repo; paper-level claims are recorded as paper evidence, not code evidence;
- InstantMesh ancestry independently shows mesh-stage RGB/mask/depth/normal + FlexiCubes regularization rather than topology-only training.

RealSaS translation:

- keep the post-M1 carrier seal;
- do not add generic adapters yet;
- if a clean authoritative target later proves undecodable, add the smallest specialist adapter/branch first;
- evaluate coarse-to-fine high-resolution extraction only after the topology/surface court shows resolution is limiting;
- evaluate sign/hole regularization only after a concrete thin-structure failure is measured;
- do not copy fixed body/hair/cloth ontology.

## 11. Quality status and thresholds

`0.1–0.2` residuals are not normalized into success because training finished.

`~0.05` remains only a historical planning reference for genuinely comparable normalized residual families. It is not a global scientific threshold.

Status vocabulary:

```text
MEASUREMENT_ONLY
OPTIMIZATION_DEBT
ELIGIBLE_FOR_COURT
COURT_PASS
```

`COURT_PASS` requires:

- authoritative target;
- held-out measurement;
- preregistered metric-specific threshold;
- no required non-regression failure.

## 12. CI state

`.github/workflows/iris_vnext_single_carrier_v1.yml` now runs on pushes to the active post-V8 branch for relevant IRIS/Compiler/docs paths and includes:

- compile checks for post-V8 policy / boundary / V8 trainer;
- quality-policy tests;
- boundary-causality tests;
- existing vNext/Compiler/native regressions.

A pushed commit is not called PASS until the corresponding workflow concludes green.

## 13. Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
4. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
5. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
6. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
7. `docs/research/IRIS_VNEXT_CURRENT_STATE_20261018.md`
8. `canonical/VNEXT_SINGLE_CARRIER_CONTRACT_V1.json`
9. `canonical/VNEXT_EXECUTION_GRAPH_V1.json`
10. `AGENTS.md`
11. `SYSTEM_INDEX.md`

Historical V2/Knight/V5/V6 artifacts are regression/provenance evidence, not the current design plan.

When the user says `continue / devam et`, continue the post-V8 quality program. Do not start another broad 12k training run, reopen the shared carrier, manufacture hidden/amodal truth, or reintroduce a second visual mesh without an explicit court changing this state.
