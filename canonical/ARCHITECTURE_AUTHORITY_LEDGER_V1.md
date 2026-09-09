# RealSaS — Architecture Authority Ledger V1

**Date:** 2026-09-09  
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
| `IRIS_SIGNED_GEOMETRY_EVIDENCE` | learned image -> signed geometry/support/uncertainty evidence | **BINDING** | `models/iris/v2/` | Neural evidence only; no final canonical IDs |
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
| `SKIN_FIELD_CODEC` | learned continuous skin-field representation/shared decoder | **BINDING MODEL ROLE; CURRENT A0 SCIENCE OPEN** | base `models/skin_field_codec/v1/`; V7 research branch is not promoted | A0 representation/decode ceiling must close before V7-native A1 |
| `ARACHNE_SKIN_PROPOSAL` | skeleton-conditioned skin/weight/deformation proposal | **BINDING ROLE; NO CURRENT V7-NATIVE FIT1 PROMOTION** | prior scaffold `models/arachne/v2/`; active research branch | Arachne owns learned proposal; Compiler owns qualification |
| `DYNAMIC_MOTION_PROOF` | qualification-owned motion probe/bake/measurement/proof | **BINDING** | current Compiler proof services/runtime interlock | Runtime/export consumes proof-owned state |

## Geppetto reference-strength FIT1 consequence

The separately preregistered reference-strength formulation closed at optimizer step `14080` with a terminal `48/48` full-structural PASS streak spanning `3072` optimizer steps.

Frozen evidence identities:

- source commit `f7be46f0a97df62a793ebf91b22297c894854f39`;
- seal commit `ae0af0cd39dd2468a012ba21890a4fed2da7c4c9`;
- checkpoint SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`;
- result SHA-256 `728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045`;
- qualified skeleton SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`.

The promoted architecture is not merely "AR1 repaired." It is a stronger formulation: lossless fieldwise surface access, exact relation message passing, global memory, prediction-only causal recurrence, per-step cross-attention, conditional diffusion, soft internal parent distribution, native STOP and separate all-pairs final parent evidence.

Therefore the old post-AR-01 state `NO_REFREEZE` is superseded for continuation. AR-01 remains valid evidence about its exact arm.

## Arachne consequence

No Arachne model is promoted by the Geppetto refreeze.

Current 2026-09-09 work is **A0 SkinFieldCodec only**, on `exp/arachne-skintokens-cleanroom-fit1-20260908`. V7 is a 278,010,880-parameter continuous forced-field codec apparatus. The current C4 experiment tests face-topology/barycentric biased dense supervision. A1 remains unauthorized until A0 closes and a V7-facing latent/decode interface plus A1 architecture/capacity contract are frozen separately.

Do not treat the existence of `models/arachne/v2/` as evidence that the current V7-native A1 has been trained or promoted.

## Historical RigAnything / AR-01 reconstruction

The earlier authoritative sequence remains historically correct:

`R2 RUN stable FAIL -> C1 RUN stable PASS -> C2 NOT RUN (staircase stop) -> C3/C4 source-only -> AR-01 separate minimal feedback RUN -> both AR-01 arms terminal FAIL -> later independent reference-strength formulation RUN -> FIT1 TERMINAL PASS -> separate promotion`

This ordering prevents two opposite mistakes:

1. falsely saying diffusion had been executed in the old Causal Repair C2 rung;
2. falsely saying diffusion still lacks any authoritative execution after the later reference-strength closure.

## Critical anti-conflation rules

1. Source exists != mechanism tested.
2. Preregistered rung != executed rung.
3. Mechanism tested != full formulation tested.
4. Historical challenger exists != current frozen Geppetto.
5. FIT1 success != generalization.
6. AR-01 != reference-strength formulation verdict.
7. A Geppetto FIT1 PASS != Arachne A0/A1 PASS.
8. A0 Codec PASS != full learned skinning closure; A1 is separate.
9. Learned proposal != Compiler canonical authority.
10. Repo prereg/result/hash authority outranks detached generated/chat drafts.

## Required update transaction

Any experiment that changes architecture belief is incomplete until reconciled across implementation/source identity, exact prereg/result, this ledger, continuation state, and explicit promotion/supersession where applicable.
