# RealSaS-OPT — Current State

**Date:** 2026-09-07  
**Canonical continuation branch:** `main`  
**Status:** `GEPPETTO_AR01_CLOSED_NO_TERMINAL_CLOSURE__POST_AR01_DIAGNOSIS_REQUIRED__FIT1_REMAINS_SCIENTIFIC_TARGET`

This file is continuation authority only on `main`.

**Live authority map:** `canonical/LIVE_AUTHORITY_MAP.md` (generated from repository refs + `canonical/AUTHORITY_MAP_V1.json`; never hand-edit)  
**Most recent closed gate:** `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE`  
**Categorical result:** `AR01_NO_TERMINAL_CLOSURE`  
**Result authority:** `canonical/AR01_RESULT_20260907.md`  
**Active experiment gate:** `NONE`  
**Next state:** `POST_AR01_DIAGNOSIS_REQUIRED__NO_NEW_EXPERIMENT_PREREGISTERED`

## One-line state

`AR-01 is closed: neither the matched no-feedback AR0 arm nor the minimal joint+parent-feedback AR1 arm satisfied the preregistered terminal 48-check structural-stability gate. AR0 nevertheless demonstrated strong Mage FIT1 reachability and a long exact streak before later collapse; AR1 learned the teacher-forced structural task but failed badly in free-running generated-state feedback, exposing a concrete exposure/recurrent-error-amplification failure mode. No Geppetto refreeze or causal winner is authorized. The next action is to choose one post-AR-01 falsifiable gate without conflating AR0 residual stability, AR1 exposure/recovery, or the already-existing fuller RigAnything C3/C4 formulation.`

## Binding authority / rehydration order

A new chat, agent, or continuation must read in this order:

1. `canonical/REHYDRATION_PACKET.md` — generated fast context packet;
2. this `CURRENT_STATE.md` — continuation / stop-go authority;
3. `canonical/CONTEXT_COVERAGE_AUDIT.md` — what history is still semantically unreconciled;
4. `canonical/LIVE_AUTHORITY_MAP.md` — generated live branch/experiment navigation view;
5. `canonical/CONTEXT_STATE_V1.json` — compact machine context;
6. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` — mechanism implementation/evidence/canonical-status map;
7. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md` — exact gate semantics and non-claims;
8. `canonical/EXPERIMENT_REGISTRY_V1.json` — experiment dependency/state graph;
9. `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` — chronological claim/decision history;
10. exact source/result artifacts referenced by those files.

No side branch becomes current truth merely because it contains newer experiments. Experimental evidence may block continuation without being promoted as product source.

## AR-01 — CLOSED

**Gate:** `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE`  
**Contract tag:** `41055bd073538d6b`  
**Closed UTC:** `2026-09-07T06:00:52.301304+00:00`  
**Verdict:** `AR01_NO_TERMINAL_CLOSURE`

The preregistered operator required each arm to finish step 16,384 with a final contiguous structural-PASS streak of at least 48 checks / 3,072 optimizer steps. A previously observed streak could not remain latched after a later failure.

### AR0 — matched feedback OFF

AR0 reached its first structural PASS at step 2,368 and later sustained a maximum 61-check exact structural streak. It nevertheless collapsed again after that region and finished step 16,384 with a structural PASS but terminal streak only 1.

Final AR0:

- outside capture = 0;
- occupancy L1 = 0;
- parent accuracy = 1.0;
- root exact = true;
- slot p95 = `0.0008499497780576348`;
- terminal contiguous structural streak = 1;
- terminal stability PASS = false.

Interpretation: same-witness reachability/capacity is strong, but no-feedback terminal optimizer/trajectory stability is still unresolved.

### AR1 — minimal skeleton-causal feedback ON

AR1 never reached a free-running structural PASS. At the final model, teacher-forced evaluation was exact while free-running evaluation remained badly outside the target region.

Final AR1 free-running:

- outside capture = 28;
- occupancy L1 = 28;
- parent accuracy = 0.50;
- root exact = true;
- slot p95 = `0.47583022713661194`;
- terminal stability PASS = false.

Final AR1 teacher-forced diagnostic:

- outside capture = 0;
- parent accuracy = 1.0;
- structural PASS = true;
- slot p95 = `0.00206305761821568`.

Teacher-forced vs free-running position max-abs gap = `0.4984188377857208`.

Interpretation: the exact minimal residual joint+parent feedback implementation is not promotable and shows a strong teacher-forced/free-running exposure/recovery failure. This is **not** a falsification of all structural autoregression.

## What AR-01 did and did not close

AR-01 **did close** the question of whether this exact minimal feedback intervention, under this exact protocol, is a terminal FIT1 repair: it is not.

AR-01 **did not** establish:

- a full RigAnything C3/C4 formulation verdict;
- that all previous-skeleton conditioning is harmful;
- that conditional diffusion is unnecessary or necessary;
- that sibling/BFS-order handling is unnecessary or necessary;
- that scheduled sampling, soft/uncertainty-aware feedback, or another recovery policy is the required fix;
- unseen-family generalization;
- Geppetto architecture refreeze.

The fuller RigAnything-mechanism challenger already exists. Read `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md` before proposing any new clean-room reimplementation.

## Post-AR-01 scientific decomposition

No new experiment is authorized yet. The next prereg must choose one clearly separated question:

1. **AR0 residual stability:** why does the no-feedback family sustain long exact streaks and then catastrophically leave the region under the current optimizer/trajectory?
2. **AR1 exposure/recovery:** why does teacher-forced structural closure fail under generated joint + hard predicted-parent feedback, and what training/feedback policy makes the generated-state process robust?
3. **Fuller formulation:** does the already-existing C3/C4 combination — including conditional diffusion and tokenized mechanical feedback/order handling — materially change the outcome under a formulation-level preregistered test?

Do not bundle these questions into a single post-hoc repair and infer causality afterward.

## What remains closed / binding

- Shipping target remains an eight-direction 2D/2.5D editable puppet; full 3D reconstruction is not product authority.
- IRIS owns learned image-to-signed-geometry evidence.
- GSA / RiggingSurfaceIR assembly, validation, provenance, normalization/packing contracts, exact legality, canonicalization and fail-close remain deterministic responsibilities.
- Compiler owns canonical IDs only after exact solve and owns final legal root/parent/tree selection.
- Geometry-only deduplication is forbidden.
- Hidden deform-node synthesis/completion remains forbidden with budget 0 in the product route.
- Teacher/source mesh at shipping inference remains forbidden.
- Lossless evidence must be preserved at the learned-consumer boundary; summary-only 24D evidence is not sufficient product authority.
- `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` remains a separate neural Geppetto-side responsibility and must not migrate into Compiler heuristics.

## FIT1 authority

FIT1 remains the immediate product-science target. FIT1 fits a generic/generalization-oriented architecture to one controlled witness; the architecture is not to be designed around Mage-specific shortcuts.

FIT1-specific optimizer interventions remain scientifically legitimate when separately preregistered, but they cannot be relabeled generic/generalization evidence. Historical `0.005` strong-trunk actual-update containment evidence remains in its own experiment lineage and is not retroactively promoted by AR-01.

AR-01 therefore does not end FIT1. It narrows the remaining Geppetto problem before refreeze.

## Geppetto refreeze / promotion rule

**No Geppetto refreeze is currently authorized.**

A future refreeze requires a new explicit decision transaction after the scientifically necessary post-AR-01 gate(s). Any promotion must atomically reconcile:

1. promoted model source;
2. source/unit/behavioral tests;
3. exact relevant experiment result + prereg/provenance hashes;
4. lossless product-conditioning consumer compatibility;
5. Compiler contract compatibility;
6. `canonical/AUTHORITY_MAP_V1.json`;
7. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`;
8. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`;
9. `canonical/EXPERIMENT_REGISTRY_V1.json`;
10. this `CURRENT_STATE.md`;
11. explicit supersession/revocation of prior architecture interpretation where applicable.

Only then create an immutable milestone tag/commit for the refrozen architecture.

## Downstream product path

Once Geppetto is genuinely refrozen:

`Geppetto closure -> Arachne FIT1 conditioned on qualified IRIS+Geppetto outputs -> SkinTokens clean-room mechanism audit/adaptation -> deformation/weight qualification -> Mage idle/breathing product witness.`

Mage breathing remains the first high-value end-to-end visible milestone; it does not itself claim cross-character generalization.

## Immediate execution order

1. preserve AR-01 result/prereg/source/runtime hashes immutably;
2. keep the AR-01 branch as evidence-only lineage;
3. finish semantic audit/backfill of Geppetto/RigAnything C0-C4 and prior stability experiments;
4. select exactly one post-AR-01 falsifiable next gate from the decomposition above;
5. preregister it before observing its scientific outcome;
6. only after required Geppetto closure/refreeze, move the product-science bottleneck to Arachne.

Until the next gate is explicitly preregistered, **do not treat AR0, AR1, current latent-only V2, or the fuller C3/C4 challenger as frozen generic Geppetto architecture authority.**