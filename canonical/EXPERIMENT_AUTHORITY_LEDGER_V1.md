# RealSaS — Experiment Authority Ledger V1

**Date:** 2026-09-09  
**Continuation authority:** `CURRENT_STATE.md` on `main`  
**Machine experiment register:** `canonical/AUTHORITY_MAP_V1.json`

This ledger records what an experiment actually tested and what it did **not** test. Branch existence or source existence is never enough to establish a result.

## Current gate matrix

| Gate / experiment | Status | Exact scientific question | What it establishes | What it does not establish |
|---|---|---|---|---|
| `GEPPETTO_REFERENCE_STRENGTH_FIT1` | **CLOSED_PASS / PROMOTED_FIT1_FROZEN** | Can the reference-strength Geppetto infer and stably retain the Mage mechanical core from current admitted IRIS/GSA evidence under native free-running generation? | Mage FIT1 terminal closure for this formulation; 48/48 terminal structural checks; separate mainline promotion | unseen-family generalization; Arachne/skin closure; PRODUCT_PASS |
| `V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION` | **ACTIVE_RUNNING** | On V7 A0, is the remaining tail driven by cancellation of active-heavy objective bias and/or missing real-face barycentric dense supervision? | after completion: causal A-vs-B and B-vs-C evidence for the exact A0 mechanisms | A1 product-time latent inference; generalization; PRODUCT_PASS |
| `V7_C3_SKINTOKENS_BOUNDARY_AWARE_DENSE_SAMPLING` | **CLOSED_FAIL** | Does a point-cloud support/near-support sampler help when BCE/MSE are corrected back to the uniform row objective with exact `u/q`? | that exact proxy/remedy did not close A0 and treatment did not beat matched control | boundary localization is false; face/barycentric source-faithful sampling is false; A1 failure |
| `V7_C2_BLEND_BOUNDARY_LOCALIZATION` | **COMPONENT_EVIDENCE** | Where do the residual V7 errors concentrate? | weak-true vs nearby-false support ordering at blend boundaries is strongly implicated | a specific remedy is necessary |
| `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE` | **CLOSED_FAIL / HISTORICAL SCOPED EVIDENCE** | Does the exact minimal AR0/AR1 joint+parent feedback intervention close Mage terminal stability? | exact AR-01 arms failed terminal closure; AR1 had severe teacher/free divergence | the later reference-strength formulation fails; all structural recurrence is harmful; generalization |
| `GEPPETTO_CAUSAL_REPAIR_V2_R2_C1` | **CLOSED_PASS / COMPONENT EVIDENCE** | Does per-step full-surface cross-attention rescue the frozen corrected-raster protocol? | R2 failed and C1 passed; xattn causally rescued that protocol | old C2 diffusion outcome; full later formulation; generalization |

## Geppetto reference-strength FIT1 — closed authority

Scientific source/optimizer commit:

`f7be46f0a97df62a793ebf91b22297c894854f39`

Seal commit:

`ae0af0cd39dd2468a012ba21890a4fed2da7c4c9`

Preregistration and frozen apparatus are preserved on main under:

`experiments/geppetto_reference_strength_fullstack_v1/`

Closure:

`canonical/GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md`

Separate promotion/refreeze:

`canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md`

Terminal result:

- categorical verdict `FIT1_TERMINAL_PASS`;
- closure step `14080`;
- terminal full-structural streak `48/48` checks = `3072` optimizer steps;
- qualified controls `22`;
- exactly one deform root;
- no teacher feedback during free-running inference;
- diffusion evaluation seeds `{11,23,47,89}`.

Frozen hashes:

- checkpoint SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`;
- final result SHA-256 `728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045`;
- qualified skeleton SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`.

The passing formulation includes lossless fieldwise surface access, exact GSA-relation message passing, global full-surface memory, prediction-only causal recurrence, per-step cross-attention, conditional residual diffusion, soft internal parent feedback, native STOP and separate all-pairs parent evidence. Canonical IDs/final legal tree remain Compiler-owned.

## AR-01 / old RigAnything staircase — anti-conflation

Historical order is:

`R2 FAIL -> C1 xattn PASS -> old staircase STOP -> AR-01 minimal feedback FAIL -> later independent reference-strength formulation FIT1 PASS -> separate promotion`.

Therefore both of these statements are simultaneously true:

1. the **old Causal Repair V2 C2 rung** was not executed because C1 passed;
2. conditional diffusion **was later executed authoritatively** inside the separately preregistered reference-strength formulation that terminal-passed FIT1.

Do not collapse those two experiments into one lineage claim.

## Arachne / SkinFieldCodec A0 — current authority

Current branch:

`exp/arachne-skintokens-cleanroom-fit1-20260908`

Current model:

- architecture `RealSaS.Arachne.SkinFieldCodec.v7`;
- `278,010,880` parameters;
- config hash `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`;
- no FSQ;
- fixed cache SHA-256 `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`;
- teacher-W SHA-256 `7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d`;
- 22 joints / 934 supervised GSA rows / 16 low-confidence rows.

### C3 closed failure

C3 preregistration:

`V7_C3_SKINTOKENS_BOUNDARY_AWARE_DENSE_SAMPLING_PREREG_20260909.md`

Frozen source Git blob:

`38541137776ae3a997e4e67b175699ac6e2fb881`

Observed terminal comparison used by C4 prereg:

- control raw row-L1 p95 `0.18175699718563249`;
- treatment raw row-L1 p95 `0.19077774486726193`;
- control raw deformation ratio `0.06118907406926155`;
- treatment raw deformation ratio `0.06876000761985779`;
- top-4 displacement `48 -> 47` only;
- combined result SHA-256 `5fe9c350e2f09a1fd858d86d39c965c9c429321c76851994379a0466d925eaa0`.

Interpretation: reject **that point-cloud + importance-corrected remedy**. Do not reject the boundary diagnosis itself.

### C4 active

Preregistration blob:

`4ada6333afa1dcfe40d08eac286a88d3acc158ad`

Source contract blob:

`cf4401f6ef3d91a32c6615b3325d9d9fbe67813b`

C4 contains three matched arms:

- A: exact C3-control importance-corrected active-only reference;
- B: exact same GSA rows without `u/q`, isolating deliberate active-heavy objective bias;
- C: real deformation-supported source faces, area-proportional face draws, triangle-interior barycentric samples and interpolated skin targets, also without importance correction.

All three use the same V7 architecture and fixed 384-step continuation protocol. The original FIT1 numerical gates remain unchanged. Top-4 remains diagnostic only.

**Authority scope:** C4 is A0 codec representation/decode science. `A1_authorized=false` remains binding. Any notebook label suggesting full Arachne/product closure must be interpreted through this narrower authority boundary.

## A0 / A1 separation

A0 training/teacher lane:

`teacher dense W -> codec representation/latent -> shared decoder -> reconstructed W`

A1 shipping learned lane:

`RiggingSurfaceIR S + QualifiedSkeletonIR G -> learned Arachne latent predictor -> same frozen codec decoder -> dense W proposal`

Therefore:

`A0 PASS -> freeze actual V7 latent/decode interface -> fresh V7-native A1 capacity/architecture prereg -> A1 FIT1 -> Compiler skin qualification -> separate product experiment`.

Neither A0 nor A1 can directly mint `PRODUCT_PASS`.

## Binding anti-conflation rules

1. Source exists != mechanism tested.
2. Scientific PASS != automatic promotion.
3. FIT1 success != generalization.
4. Geppetto PASS != Arachne PASS.
5. A0 Codec PASS != A1 inference PASS.
6. A1 PASS != PRODUCT_PASS.
7. Compiler qualification cannot silently replace missing learned semantics.
8. Repo prereg/result/hash authority outranks detached chat/generated wording.
9. Routine Actions/science execution uses only the local self-hosted RealSaS runner.

## Promotion transaction rule

A closure that changes current authority is incomplete until source identity, exact prereg/result, architecture/experiment ledgers, machine context, `CURRENT_STATE.md`, provenance/supersession and a separate promotion decision are reconciled.
