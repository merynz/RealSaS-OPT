# RealSaS-OPT — Current State

**Date:** 2026-10-18  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Review vehicle:** draft PR #116, to be fast-forwarded to the active source head  
**Active program:** **POST-V8 QUALITY / TRUTH / CONSUMPTION QUALIFICATION**  
**Scientific status:** **NO IRIS FIT8 SCIENTIFIC PASS**

## Authority separation

Do not collapse these into one notion of “current”:

1. `main` is merged production-code authority.
2. The completed V8 Fit8 run is the architectural/measurement baseline, bound to source commit `5083c9915533ec8ec6530c7d229c635887abfea7` plus its exact external checkpoint/receipt lineage.
3. The active post-V8 branch contains future-source hardening and audits. It does not retroactively mutate the completed V8 artifact.
4. Artifact Registry / Attempt / ProductRevision remain durable research/product authority. Git chronology is not a substitute.

## Training lineage

### V5 — shared-H single-objective training falsified

A1 learned appearance while mechanics and later surface authority collapsed. A downstream objective can improve its own metric while rewriting useful shared representation.

### V6 — cumulative retention only partially worked

Prior-objective retention preserved mechanics much better but did not preserve geometry/extraction through A1. Loss balancing is therefore not sufficient carrier protection.

### V8 — post-M1 carrier isolation supported

V8 migrated from the exact V6 M1-end state and sealed `scene_core` for downstream specialist stages. It reached step 12000.

Across step3600→12000 the recorded geometry/extraction baseline stayed invariant while appearance, current relation targets and topology objective continued learning.

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

This is structural isolation, not temporary loss balancing.

V8 is a **measurement baseline**, not a scientific Fit8 PASS: metric-specific preregistered thresholds and an independent topology court were absent.

## Executable carrier hardening

`models/iris/vnext/quality_policy_v1.py` now uses exact downstream V8 stage IDs:

```text
A1_APPEARANCE_EVIDENCE
R1_ROLE_FREE_RELATIONS
T1_MECHANICAL_STRUCTURE
J1_JOINT_FINETUNE
```

Unknown A/R/T/J aliases fail closed.

Future `IRISVNextFit8TrainerV8` execution validates both configured `scene_lr_scale` and the actual optimizer `scene_core` group LR in `_apply_stage_lrs()`. A post-M1 non-zero scene-core LR requires an explicit reopening court.

The historical completed V8 artifact remains authority-bound to `5083...`; this hardening governs future source execution.

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
fragment order  contact/attachment
 graph   graph      graph
        ↓
 one canonical M/G/W_M
        ↓
     Runtime
```

No mandatory named body-part taxonomy exists in the inference contract. No independent deformable visual mesh is allowed.

## What frozen H currently establishes

Under current teacher definitions, sealed H supports learning of appearance, same-drawable proxy, attachment proxy/evidence, front/back, occlusion, visibility and topology objective.

This is **decodability evidence**, not proof that every teacher target is semantically authoritative.

Near-zero relation residual must not be promoted to “graphics solved” without teacher, held-out and runtime courts.

## Highest-priority truth debt: boundary / decomposition causality

Boundary evidence may reflect geometric termination, material transition, alpha transition, occlusion, source seam, drawable ownership or intentional presentation logic. Current generic FIT8 assets do not uniquely identify all of those.

`BoundaryCausalityTeacherV1` admits only provable causes and uses `RESIDUAL_UNKNOWN` as abstention. Unknown emits no training target.

### Historical relation producer correction

There are two distinct V1-era paths and they must not be conflated:

```text
models/iris/vnext/graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

active Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

Therefore V8's cut metric is about the historical **owner-or-material boundary proxy**, not simply `1 - same_drawable`, and not authoritative drawable-cut truth.

Exact audit: `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`.

### Replacement already started

New contracts:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

V2 requires explicit value + authority + provenance + `valid_for_training` for each relation target. Missing truth abstains. The new loss requires explicit per-family/sample validity masks and refuses implicit all-negative supervision.

V2 is **not yet a populated corpus/training authority**. Producers and support statistics must be wired before a focused run.

## StdGEN++ foundation branch disposition

Research provenance branch:

`agent/unified-iris-stdgenpp-foundation` @ `d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`

Salvage source-defensible evidence:

- geometry/topology provenance;
- authored material provenance;
- rig/skin/deformation evidence;
- all-hit ray stacks;
- fixed-frame visibility/order;
- geometric front/occlusion/visibility;
- component/residual-boundary diagnostics;
- local continuation as evidence.

Do not promote from generic FIT8 3D alone:

- authoritative drawable owner;
- hidden/amodal support;
- intentional 2D presentation override;
- global named semantic identities;
- inferred hidden continuation as ground truth.

The stronger “full owner/amodal presentation truth is recoverable from these generic 3D assets” program was falsified by identifiability, not just implementation difficulty.

## Compiler/runtime consumption

Machine-readable audit:

`docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`

Current source evidence:

- same/cut/boundary influence fragment clustering;
- front + occlusion gate directed order edges;
- native order edges can override contradictory physical depth;
- visibility controls native `face_visible()` and a native E2E proves Graphics visibility changes final RGBA;
- attachment records survive Compiler → RuntimeGraphics → `graphics.bin`, but the current native reference raster does not use them behaviorally;
- support-band pixels survive serialization/validation but are not applied by the reference renderer.

Serialized is not the same as behaviorally used.

## TESSA/topology state

V8 showed topology-objective learnability from sealed H but did not seal an independent topology PASS.

Required held-out court must measure actual topology/surface outcomes, including valid/degenerate faces, component error, boundary loops/holes, non-manifold behavior where meaningful, canonical-surface deviation, source support and held-out stability.

Training loss is not topology qualification.

## StdGEN++ re-audit

Detailed dossier:

`docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`

Current lessons:

- public StdGEN code verifies constrained LoRA transformer adaptation plus trainable synthesizer/decoders;
- SDF/RGB/deformation/FlexiCubes/semantics share triplane features;
- semantics actively affect equivalent-SDF extraction, so their functional role must be replaced, not merely their class names deleted;
- StdGEN++ paper adds full-body/facial specialization, three supervision stages, multilayer truth, coarse-to-fine high-resolution extraction and hole regularization;
- exact StdGEN/StdGEN++ training-loop source is not public in the StdGEN repository; paper evidence and code evidence remain separately labeled;
- InstantMesh ancestry independently demonstrates mesh-stage RGB/mask/depth/normal + FlexiCubes regularization.

RealSaS translation:

- keep the post-M1 carrier seal;
- do not add generic adapters yet;
- if a clean authoritative target later proves undecodable, modify the specialist first;
- evaluate high-resolution active extraction only if topology/surface courts show resolution is limiting;
- evaluate sign/hole regularization only after measured thin-structure failure;
- do not copy a fixed body/hair/cloth ontology.

## Quality / PASS discipline

`0.1–0.2` residuals are not normalized into success because training finished.

`~0.05` is only a historical planning reference for genuinely comparable normalized residual families, not a global scientific threshold.

```text
MEASUREMENT_ONLY
OPTIMIZATION_DEBT
ELIGIBLE_FOR_COURT
COURT_PASS
```

`COURT_PASS` requires authoritative target + held-out metric + preregistered metric-specific threshold + required non-regression.

## CI state

`.github/workflows/iris_vnext_single_carrier_v1.yml` now runs on relevant pushes to the active branch and gates:

- post-V8 policy/trainer compilation;
- carrier-seal tests;
- Boundary Causality V1;
- Graphics Truth V2 explicit-validity contracts;
- existing vNext/Compiler/native regressions.

A pushed commit is not called PASS until its workflow concludes green.

## Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
4. `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`
5. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
6. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
7. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
8. `canonical/VNEXT_SINGLE_CARRIER_CONTRACT_V1.json`
9. `canonical/VNEXT_EXECUTION_GRAPH_V1.json`
10. `AGENTS.md`
11. `SYSTEM_INDEX.md`

When the user says `continue / devam et`, continue the post-V8 quality program. Do not launch another broad 12k run, reopen the shared carrier, manufacture hidden/amodal truth, or revive a second visual mesh without an explicit court changing this state.
