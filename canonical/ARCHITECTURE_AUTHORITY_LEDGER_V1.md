# RealSaS — Architecture Authority Ledger V1

**Date:** 2026-09-10  
**Scope:** cross-chat / cross-agent architecture reconstruction  
**Continuation authority:** `CURRENT_STATE.md` on `main`

This ledger does **not** independently authorize promotion. It answers: **for each important mechanism, where is it implemented, what has actually been executed/tested, and is it current product/FIT1 architecture or only research lineage?**

## Status vocabulary

- `BINDING` — current responsibility/invariant on `main`.
- `FIT1_FROZEN` — separately promoted formulation that is current for the controlled Mage FIT1 witness; generalization remains unclaimed.
- `IMPLEMENTED_RESEARCH` — executable research mechanism exists; no current promotion authority.
- `TESTED_COMPONENT` — controlled/diagnostic evidence exists for the named mechanism only.
- `CLOSED_REJECTED_IMPLEMENTATION` — exact implementation failed its gate; broader mechanism family may remain open.
- `OPEN_REQUIREMENT` — unresolved required responsibility/diagnosis.
- `FORBIDDEN` — prohibited product behavior.

## Current authority matrix

| ID | Mechanism / responsibility | Status | Implementation / evidence | Canonical interpretation |
|---|---|---|---|---|
| `SYS_V4_2D_PUPPET_TARGET` | 8-direction editable 2D/2.5D puppet; not full 3D reconstruction | **BINDING** | `canonical/SYSTEM_ARCHITECTURE_V4_20260902.md` | Product target remains 2D appearance + editable rig/deformation/runtime |
| `IRIS_V2_OBSERVATION_FOUNDATION_EVIDENCE` | promoted observation/foundation/evidence stack used by the current IRIS lineage | **BINDING BASE LAYER** | `models/iris/v2/`; restoration source audit | V2 remains the promoted observation/foundation package; it is not the whole current signed-geometry composition |
| `IRIS_SIGNED_GEOMETRY_EVIDENCE` | scene-first all-view signed geometry/support/uncertainty evidence for current Mage FIT1 witness | **BINDING / PROMOTED MAGE FIT1 WITNESS** | `models/iris/v3/`; `models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json`; self-hosted promotion gate | V3 is the current promoted signed-field head/witness; neural evidence only, GSA owns `RiggingSurfaceIR`, no final canonical IDs |
| `GSA_RIGGING_SURFACE_ASSEMBLY` | evidence -> typed `RiggingSurfaceIR`, packing/validation/provenance | **BINDING** | current Compiler substrate / surface assembly | Deterministic contract owner |
| `LOSSLESS_LEARNED_EVIDENCE_BOUNDARY` | learned consumer retains lossless per-view surface/raster/support evidence | **BINDING / PROMOTED FOR GEPPETTO FIT1** | `models/geppetto/reference_strength_v1/rigging_surface_tensorization_v1.py` + sealed FIT1 source | Summary-only historical 24D conditioning is not current Geppetto FIT1 authority |
| `GEPPETTO_REFERENCE_STRENGTH_V1` | current FIT1-frozen learned skeleton/control proposal formulation | **FIT1_FROZEN** | `models/geppetto/reference_strength_v1/`; promotion `canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md` | Mage FIT1 terminal PASS; generalization not claimed |
| `GEPPETTO_V2_LATENT_RECURRENCE` | prior latent-state recurrent proposal decoder | **SUPERSEDED_FOR_FIT1 / PRESERVED** | `models/geppetto/v2/` | Prior mainline/research provenance; not the frozen FIT1 formulation |
| `FULL_SURFACE_PER_STEP_XATTN` | every generation step can attend full surface memory | **FIT1_FROZEN / EXECUTED** | reference-strength V1 + prior Causal Repair C1 evidence | Component was previously isolated; now also present in the terminal-PASS frozen formulation |
| `CONDITIONAL_DIFFUSION_LOCUS` | conditional diffusion residual refinement of continuous locus | **FIT1_FROZEN / EXECUTED** | `GeppettoReferenceStrengthNoLearnedSlotV1`; final four-seed evaluation `{11,23,47,89}` | Old Causal Repair staircase never ran C2, but later reference-strength FIT1 did execute diffusion inside the passing full formulation |
| `PREDICTION_ONLY_CAUSAL_CONTROL_RECURRENCE` | generated-state causal control recurrence with no teacher geometry/parent feedback in free run | **FIT1_FROZEN / EXECUTED** | reference-strength V1 prereg + closure | Distinct from the failed AR-01 minimal residual feedback arm |
| `SOFT_INTERNAL_PARENT_FEEDBACK` | learned soft distribution over previous generated controls used as internal recurrent mechanical context | **FIT1_FROZEN / EXECUTED** | reference-strength V1 | Final canonical parent/tree remains Compiler-owned |
| `NATIVE_STOP_COUNT` | model decides generated control count through native STOP within resource guard | **FIT1_FROZEN / EXECUTED** | closure step 14080, qualified count 22 | `22` is target outcome, not an architectural product cap |
| `DETERMINISTIC_CANONICAL_VIEW_DIRECTION` | fixed Fourier code from exact eight 45° yaw camera directions | **FIT1_FROZEN / BINDING** | `GeppettoReferenceStrengthNoLearnedSlotV1` | No learned absolute V0..V7 slot embedding in frozen formulation |
| `RIGANYTHING_FORMULATION_CHALLENGER` | older clean-room full-surface/diffusion/mechanical/order research challenger | **IMPLEMENTED_RESEARCH / HISTORICAL MECHANISM LINEAGE** | `models/geppetto/challengers/riganything_mechanisms_v1.py`, `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md` | Do not confuse exact old challenger rungs with the separately frozen reference-strength formulation |
| `SKELETON_CAUSAL_MECHANICAL_FEEDBACK_AR01` | minimal residual joint+parent recurrent feedback tested by AR-01 | **CLOSED_REJECTED_IMPLEMENTATION / TESTED_COMPONENT** | `canonical/AR01_RESULT_20260907.md` | Exact AR-01 implementation failed; does not contradict the later reference-strength PASS because the formulations differ materially |
| `GEPPETTO_PARENT_EVIDENCE` | neural parent/root likelihood/evidence | **BINDING RESPONSIBILITY** | reference-strength all-pairs parent evidence + Compiler qualifier | Compiler owns legal final parent/root selection |
| `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` | whether controls are necessary/simplifiable | **BINDING GEPPETTO EVIDENCE ROLE / BROADER GENERALIZATION OPEN** | reference-strength salience head | Never migrate semantic cleanup into Compiler legality heuristics |
| `COMPILER_EXACT_GRAPH_LEGALITY` | final legal graph/root/parent/tree solve | **BINDING** | current Compiler exact qualification | Deterministic legality; does not invent semantic evidence |
| `CANONICAL_IDS_POST_SOLVE` | canonical identity after exact solve | **BINDING** | current Compiler | Neural model does not own final IDs |
| `GEOMETRY_ONLY_DEDUP` | geometry-based proposal identity contraction | **FORBIDDEN** | n/a | Preserve identities absent qualified neural relation evidence |
| `HIDDEN_DEFORM_NODE_COMPLETION` | silently add missing deform controls | **FORBIDDEN (budget 0)** | n/a | Compiler cannot repair cardinality by synthesis |
| `SKIN_FIELD_CODEC` | learned continuous skin-field representation/shared decoder | **BINDING MODEL ROLE / A0 OPEN AFTER C4 FAIL** | base `models/skin_field_codec/v1/`; V7 research branch; `canonical/ARACHNE_A0_V7_C4_CLOSURE_20260910.md` | V7 continuous field identity survives, but C4 rejected the tested sampling family and A0 has not closed; next requirement is separately preregistered within-support calibration diagnosis |
| `ARACHNE_C4_OBJECTIVE_BIAS_INTERVENTION` | remove exact `u/q` correction while holding A/B queries and prefix fixed | **CLOSED_REJECTED_IMPLEMENTATION / TESTED_COMPONENT** | C4 A vs B | Unit-weight active-heavy objective worsened p95/deformation; objective-bias cancellation is not supported as the blocker |
| `ARACHNE_C4_FACE_BARY_DENSE_SUPERVISION` | real deformation-supported faces + area draw + triangle-interior barycentric target supervision | **CLOSED_REJECTED_IMPLEMENTATION / TESTED_COMPONENT** | C4 B vs C | Exact source-face/barycentric port worsened primary metrics sharply and degraded ownership; this does not falsify all topology-aware methods |
| `ARACHNE_WITHIN_SUPPORT_CALIBRATION_DIAGNOSIS` | isolate remaining blend-ratio/calibration failure inside true support | **OPEN_REQUIREMENT / NOT YET PREREGISTERED** | required next fork from C4 prereg/closure | Diagnostic direction only; no custom loss/treatment is authorized yet and frozen 0.05 gates remain unchanged |
| `ARACHNE_SKIN_PROPOSAL` | skeleton-conditioned skin/weight/deformation proposal | **BINDING ROLE; NO CURRENT V7-NATIVE FIT1 PROMOTION** | prior scaffold `models/arachne/v2/`; active lineage branch retained as evidence only | Arachne owns learned proposal; Compiler owns qualification; A1 remains blocked until A0 closes |
| `DYNAMIC_MOTION_PROOF` | qualification-owned motion probe/bake/measurement/proof | **BINDING** | current Compiler proof services/runtime interlock | Runtime/export consumes proof-owned state |

## IRIS layered-current consequence

The current FIT1 upstream witness must be read as one ownership envelope, not as a filename-version contest:

`8 RGBA + exact cameras -> promoted V2 observation/foundation evidence where required by the lineage -> promoted V3 scene-first all-view signed field -> deterministic GSA compaction/local geometry/provenance -> RiggingSurfaceIR`.

The promoted Mage V3 witness binds IRIS checkpoint SHA-256 `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2` and signed zero-surface SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`; teacher mesh is not used at inference. V3 therefore supersedes the old shorthand “IRIS current = V2 only” for signed-geometry FIT1 composition, while V2 remains a promoted supporting source layer.

## Geppetto reference-strength FIT1 consequence

The separately preregistered reference-strength formulation closed at optimizer step `14080` with a terminal `48/48` full-structural PASS streak spanning `3072` optimizer steps.

Frozen evidence identities:

- source commit `f7be46f0a97df62a793ebf91b22297c894854f39`;
- seal commit `ae0af0cd39dd2468a012ba21890a4fed2da7c4c9`;
- checkpoint SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`;
- result SHA-256 `728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045`;
- qualified skeleton SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`;
- machine evidence manifest `canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`.

The promoted architecture is not merely "AR1 repaired." It is a stronger formulation: lossless fieldwise surface access, exact relation message passing, global memory, prediction-only causal recurrence, per-step cross-attention, conditional diffusion, soft internal parent distribution, native STOP and separate all-pairs final parent evidence.

Therefore the old post-AR-01 state `NO_REFREEZE` is superseded for continuation. AR-01 remains valid evidence about its exact arm.

## Arachne consequence after C4

No Arachne model is promoted by the Geppetto refreeze or by C4.

Current A0 apparatus is `RealSaS.Arachne.SkinFieldCodec.v7`, `278,010,880` parameters, continuous forced-field transport, no FSQ. C4 is closed scientific evidence, not an active experiment.

C4 closure authority:

`canonical/ARACHNE_A0_V7_C4_CLOSURE_20260910.md`

Evidence manifest:

`canonical/ARACHNE_A0_V7_C4_EVIDENCE_MANIFEST_V1.json`

Research-branch closure commit:

`d0b666725e79f3beb0ea001f456375421cc14375`

Aggregate result SHA-256:

`439583ac60eca855dc2b55efabf5b4a6df4de5f9617e9e35583cf136b4cb55d9`

C4 established that the exact A/B objective-bias intervention and B/C source-face/barycentric intervention do not close the A0 gate. B worsened relative to A; C worsened sharply relative to B and degraded dominant ownership. The non-accuracy legality/numerical guards remained healthy, so the result is scientific FAIL rather than infrastructure failure.

The architecture consequence is intentionally narrow:

- retain the V7 continuous field representation as the current A0 research apparatus;
- reject **this exact sampling family** as the next repair;
- retain the prior blend-boundary diagnosis as unresolved rather than declaring it false;
- require a separately preregistered **within-support blend-ratio/calibration diagnosis** before another A0 optimizer step;
- do not auto-stack a custom calibration loss;
- do not change the frozen `0.05` FIT1 gates;
- keep A1 unauthorized until a future A0 terminal PASS freezes the actual V7 latent/decode interface.

Do not treat the existence of `models/arachne/v2/` as evidence that the current V7-native A1 has been trained or promoted.

## Historical RigAnything / AR-01 reconstruction

The earlier authoritative sequence remains historically correct:

`R2 RUN stable FAIL -> C1 RUN stable PASS -> C2 NOT RUN (staircase stop) -> C3/C4 source-only in that old Geppetto challenger -> AR-01 separate minimal feedback RUN -> both AR-01 arms terminal FAIL -> later independent reference-strength formulation RUN -> FIT1 TERMINAL PASS -> separate promotion`.

This Geppetto history is distinct from the later Arachne V7 experiments that also use labels C2/C3/C4. Do not conflate experiment families by suffix alone.

## Critical anti-conflation rules

1. Source exists != mechanism tested.
2. Preregistered rung != executed rung.
3. Mechanism tested != full formulation tested.
4. Historical challenger exists != current frozen Geppetto.
5. FIT1 success != generalization.
6. AR-01 != reference-strength formulation verdict.
7. Geppetto FIT1 PASS != Arachne A0/A1 PASS.
8. A0 Codec science != A1 product-time latent inference.
9. C4 A0 FAIL != A1 or product failure.
10. C4 rejection of one sampling family != rejection of every topology-aware method.
11. Learned proposal != Compiler canonical authority.
12. Repo prereg/result/hash authority outranks detached generated/chat drafts.

## Required update transaction

Any experiment that changes architecture belief is incomplete until reconciled across implementation/source identity, exact prereg/result, this ledger, continuation state, and explicit promotion/supersession where applicable. C4 now satisfies that requirement only after the matching main authority reconciliation; it promotes no Arachne model.
