# RealSaS — Architecture Authority Ledger V1

**Date:** 2026-09-07  
**Scope:** cross-chat / cross-agent architecture reconstruction  
**Continuation authority:** `CURRENT_STATE.md` on `main`

This ledger does **not** independently authorize promotion. It answers: **for each important mechanism, where is it implemented, what has actually been executed/tested, and is it current product architecture or only research lineage?**

## Status vocabulary

- `BINDING` — current responsibility/invariant on `main`.
- `MAINLINE_ON_HOLD` — source exists but refreeze/promotion is blocked.
- `IMPLEMENTED_RESEARCH` — executable research mechanism exists; no product authority.
- `TESTED_COMPONENT` — controlled/diagnostic evidence exists for the named mechanism only.
- `CLOSED_REJECTED_IMPLEMENTATION` — exact implementation failed its gate; broader mechanism family may remain open.
- `OPEN_REQUIREMENT` — unresolved required responsibility/diagnosis.
- `FORBIDDEN` — prohibited product behavior.

## Current authority matrix

| ID | Mechanism / responsibility | Status | Implementation / evidence | Canonical interpretation |
|---|---|---|---|---|
| `SYS_V4_2D_PUPPET_TARGET` | 8-direction editable 2D/2.5D puppet; not full 3D reconstruction | **BINDING** | `canonical/SYSTEM_ARCHITECTURE_V4_20260902.md` | Product target remains 2D appearance + editable rig/deformation/runtime |
| `IRIS_SIGNED_GEOMETRY_EVIDENCE` | learned image -> signed geometry/support/uncertainty evidence | **BINDING** | `models/iris/v2/` | Neural evidence only; no final canonical IDs |
| `GSA_RIGGING_SURFACE_ASSEMBLY` | evidence -> typed RiggingSurfaceIR, packing/validation/provenance | **BINDING** | current Compiler substrate / surface assembly | Deterministic contract owner |
| `LOSSLESS_LEARNED_EVIDENCE_BOUNDARY` | learned consumer retains lossless per-view evidence | **BINDING REQUIREMENT; CONSUMER PROMOTION UNRESOLVED** | V3P/V3X/AR-01 lineage | Summary-only 24D is not sufficient product authority |
| `GEPPETTO_V2_LATENT_RECURRENCE` | current mainline latent-state recurrent proposal decoder | **MAINLINE_ON_HOLD** | `models/geppetto/v2/geppetto_candidate_v2.py` | Substantial Mage capacity exists, but no generic refreeze is authorized |
| `FULL_SURFACE_PER_STEP_XATTN` | every generation step can attend full surface memory | **TESTED_COMPONENT / IMPLEMENTED_RESEARCH** | Causal Repair V2 contract `233ec3bcd77e0f02`: R2 FAIL -> C1 xattn PASS; V3X/AR-01 also consume full surface | Controlled evidence that xattn rescued that frozen Mage protocol; not theoretical necessity or full formulation equivalence |
| `CONDITIONAL_DIFFUSION_LOCUS` | conditional diffusion continuous locus | **IMPLEMENTED_RESEARCH — NOT AUTHORITATIVELY RUN IN CAUSAL REPAIR V2** | `ConditionalJointDiffusionV1` and C2 rung exist; C1 PASS stopped the prereg staircase before C2 | Do not describe C2 as tested from Causal Repair V2; diffusion remains an open research option |
| `RIGANYTHING_FORMULATION_CHALLENGER` | full-surface + diffusion + joint/parent feedback token + sibling/BFS helper | **IMPLEMENTED_RESEARCH** | creation commit `ba634955777479ee05a5b199742710b1736123b5`, blob `136d358572be6ff72ba15999474b6d3be600f5bc`; `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md` | Fuller challenger already exists; no authoritative C3/C4 full-formulation result registered |
| `SKELETON_CAUSAL_MECHANICAL_FEEDBACK_AR01` | minimal residual joint+parent recurrent feedback | **CLOSED_REJECTED_IMPLEMENTATION / TESTED_COMPONENT** | `canonical/AR01_RESULT_20260907.md`, contract `41055bd073538d6b` | Exact AR-01 implementation not promotable; no full-AR/formulation falsification |
| `AR1_GENERATED_STATE_EXPOSURE_RECOVERY` | robustness to generated joint + predicted-parent state | **OPEN_REQUIREMENT / DIAGNOSIS** | AR1 final teacher-forced exact vs free-running 28 outside / parent .50 | Future policy experiment may test scheduled sampling/soft/uncertain feedback, but none is established as necessary |
| `AR0_TERMINAL_TRAJECTORY_STABILITY` | prevent late catastrophic departures after long exact streaks | **OPEN_REQUIREMENT / DIAGNOSIS** | AR0 max 61 exact checks but terminal streak 1 | Strong reachability survives; terminal optimizer/trajectory stability remains open |
| `SIBLING_BFS_ORDER_AUGMENTATION` | equivalent sibling/depth ordering augmentation | **IMPLEMENTED_RESEARCH; NOT CLOSED** | C4 helper/source lineage | Source existence is not tested/promoted evidence |
| `GEPPETTO_PARENT_EVIDENCE` | neural parent/root likelihood/evidence | **BINDING RESPONSIBILITY** | Geppetto proposal side; AR-01 strict causal parent head diagnostic | Compiler still owns legal final parent/root selection |
| `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` | whether controls are necessary/simplifiable | **OPEN_REQUIREMENT** | no promoted solution | Neural Geppetto-side responsibility; never Compiler heuristic cleanup |
| `COMPILER_EXACT_GRAPH_LEGALITY` | final legal graph/root/parent/tree solve | **BINDING** | current Compiler exact qualification | Deterministic legality; does not invent semantic evidence |
| `CANONICAL_IDS_POST_SOLVE` | canonical identity after exact solve | **BINDING** | current Compiler | Neural model does not own final IDs |
| `GEOMETRY_ONLY_DEDUP` | geometry-based proposal identity contraction | **FORBIDDEN** | n/a | Preserve identities absent qualified neural relation evidence |
| `HIDDEN_DEFORM_NODE_COMPLETION` | silently add missing deform controls | **FORBIDDEN (budget 0)** | n/a | Compiler cannot repair cardinality by synthesis |
| `SKIN_FIELD_CODEC` | learned continuous skin-field representation/shared decoder | **BINDING CURRENT MODEL ROLE** | `models/skin_field_codec/v1/` | Learned proposal representation only |
| `ARACHNE_SKIN_PROPOSAL` | skeleton-conditioned skin/weight/deformation proposal | **BINDING CURRENT MODEL ROLE; FUTURE SCIENCE OPEN** | `models/arachne/v2/` | Arachne owns learned proposal; Compiler owns qualification |
| `DYNAMIC_MOTION_PROOF` | qualification-owned motion probe/bake/measurement/proof | **BINDING** | current Compiler proof services/runtime interlock | Runtime/export consumes proof-owned state |

## Exact RigAnything-related execution reconstruction

The authoritative sequence is now:

`R2 RUN stable FAIL -> C1 RUN stable PASS -> C2 NOT RUN (staircase stop) -> C3/C4 source-only -> AR-01 separate minimal feedback RUN -> both AR-01 arms terminal FAIL`

Causal Repair V2 C1 exact evidence:

- contract `233ec3bcd77e0f02`;
- first stable step 4992;
- best slot p95 `0.004098494071513414` at 6208;
- final outside 0;
- final slot p95 `0.005874851252883673`;
- recorded interpretation: xattn rescues **this frozen training protocol**, not theoretical necessity.

This explicitly revokes any earlier shorthand implying C2 was experimentally executed in that staircase.

## AR-01 architecture consequence

AR-01 contract `41055bd073538d6b` closed `AR01_NO_TERMINAL_CLOSURE`:

1. Do not promote the minimal residual feedback implementation.
2. Do not refreeze latent-only AR0 merely because it reached long exact streaks; terminal stability failed.
3. Do not infer full RigAnything failure; C2 was not in AR-01, C4 ordering was absent, and C3/C4 remain source-only.
4. Do not move mechanical salience/cleanup into Compiler.

Current architecture state:

`NO_REFREEZE__SEPARATE_AR0_STABILITY_FROM_AR1_EXPOSURE_AND_FULLER_FORMULATION_QUESTION`

## Critical anti-conflation rules

1. Source exists != mechanism tested.
2. Preregistered rung != executed rung.
3. Mechanism tested != full formulation tested.
4. Research challenger exists != canonical Geppetto contains it.
5. FIT1 success != generalization.
6. AR-01 != RigAnything formulation verdict.
7. AR0 long exact streak != terminal PASS.
8. AR1 failure != all structural AR falsified.
9. Teacher-forced closure != free-running closure.
10. Before saying “RealSaS lacks X”, inspect this ledger + lineage dossier + listed source, not only mainline.
11. Repo prereg/result hashes outrank detached generated drafts.

## Required update transaction

Any experiment that changes architecture belief is incomplete until reconciled across:

1. implementation commit/blob;
2. exact prereg/result pointer;
3. this ledger;
4. experiment authority ledger;
5. experiment registry;
6. compact context state;
7. CURRENT_STATE if stop/go changes;
8. append-only scientific journal;
9. explicit supersession/revocation where applicable.

Do not create a competing mechanism ledger; extend or explicitly supersede this one.