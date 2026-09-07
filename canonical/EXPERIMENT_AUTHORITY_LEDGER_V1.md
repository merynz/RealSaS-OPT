# RealSaS — Experiment Authority Ledger V1

**Date:** 2026-09-07  
**Continuation authority:** `CURRENT_STATE.md` on `main`  
**Machine experiment register:** `canonical/AUTHORITY_MAP_V1.json`  
**Structured dependency registry:** `canonical/EXPERIMENT_REGISTRY_V1.json`

This ledger records what an experiment actually tested and what it did **not** test. The live branch/head/status view is generated in `canonical/LIVE_AUTHORITY_MAP.md`; do not copy live branch SHAs here.

## Status vocabulary

- `ACTIVE_RUNNING` — preregistered experiment has not reached its final fixed verdict.
- `CLOSED_PASS` / `CLOSED_FAIL` — exact preregistered gate closed.
- `COMPONENT_EVIDENCE` — mechanism-level evidence only.
- `IMPLEMENTED_NOT_AUTHORITATIVELY_RUN` — executable/preregistered source exists but does not constitute a scientific result.
- `OPEN_NOT_RUN` — recognized gate/question with no closed run.
- `INDEX_GAP_NEEDS_BACKFILL` — historical evidence exists, but exact canonical result pointer must be reverified before numerical claims are relied on.

## Current gate matrix

| Gate / experiment | Status | Exact scientific question | What it can establish | What it cannot establish |
|---|---|---|---|---|
| `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE` | **CLOSED_FAIL — `AR01_NO_TERMINAL_CLOSURE`** | Matched AR0 vs AR1: does minimal joint+parent mechanical recurrent feedback improve Mage FIT1 reachability/retention and parent/root structural prediction? | Neither arm closed the terminal gate; AR0 has strong reachability with residual late collapse; AR1 shows severe teacher/free exposure failure | Full RigAnything verdict; all AR harmful; a particular recovery method necessary; generalization; promotion |
| `POST_AR01_DIAGNOSIS_NEXT_GATE_SELECTION` | **OPEN_NOT_RUN** | Which remaining uncertainty should be isolated next: AR0 residual stability, AR1 exposure/recovery, or fuller C3/C4 behavior? | A future prereg can isolate one of these questions | Selection itself proves no mechanism |
| `GEPPETTO_CAUSAL_REPAIR_V2_R2_C1` | **CLOSED_PASS — C1 winner** | On corrected-raster conditioning, which minimal staircase rung first stably fits Mage: R2, C1 xattn, then C2 diffusion only if needed? | R2 failed stable gate; C1 full-surface xattn passed under the frozen protocol, so the staircase stopped | C2 diffusion behavior; skeleton feedback; full C3/C4; theoretical necessity; generalization |
| `GEPPETTO_C2_CONDITIONAL_DIFFUSION_LOCUS` | **IMPLEMENTED_NOT_AUTHORITATIVELY_RUN** | Add conditional diffusion on top of C1, with geometry feedback absent | Preserved implementation/prereg design only | No C2 outcome can be claimed from Causal Repair V2 because C1 PASS stopped the staircase before C2 |
| `RIGANYTHING_FULL_FORMULATION_CHALLENGER_C3_C4` | **IMPLEMENTED_NOT_AUTHORITATIVELY_RUN** | Full-surface + diffusion + generated/teacher joint/parent feedback token, with sibling/BFS helper | Existing fuller formulation-level implementation/design lineage | Source existence is not evidence; C1 + AR-01 do not sum into C3/C4 verdict |
| `V3P_LOSSLESS_EVIDENCE_STABILITY` | **COMPONENT_EVIDENCE** | Preserve lossless fixed 8-view evidence side-path | Lossless consumer-seam behavior and same-witness diagnostics | Generic promotion/generalization |
| `V3X_FULL_SURFACE_STABILITY` | **COMPONENT_EVIDENCE** | V3P + per-step full-surface access | Reachability/stability diagnostic | Structural AR or formulation closure |
| `FIT1_TRUNK_ACTUAL_UPDATE_CAP_0_005` | **COMPONENT_EVIDENCE** | FIT1-specific strong-trunk update containment | Mage containment evidence under preregistered use | Generic architecture stability/generalization/product-optimal threshold |
| `MAGE_FIT1_HISTORICAL_B1S` | **INDEX_GAP_NEEDS_BACKFILL** | Historical same-witness fit of latent-only family | Same-witness capacity after exact pointer normalization | Structural-AR necessity/generalization |
| `R6_COVERAGE_STRESS` | **INDEX_GAP_NEEDS_BACKFILL** | Coverage stress | Coverage evidence after exact pointer normalization | FIT1/AR/generalization closure |
| `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` | **OPEN_NOT_RUN** | Learn whether deform controls are mechanically necessary/simplifiable | Future neural responsibility closure | Compiler legality cannot substitute for semantics |

## Causal Repair V2 exact authority

**Drive folder:** `RealSaS_MAGE_GEPPETTO_CAUSAL_REPAIR_V2_NO_TOKEN`  
**Contract:** `233ec3bcd77e0f02`  
**Comparison created:** `2026-09-06T04:06:17.337Z`

Preregistered staircase:

`R2 -> if FAIL C1 -> if C1 FAIL C2`

Authoritative runs folder contains **R2 and C1 only**.

### R2

- stable PASS: false
- best slot p95: `0.041321732103824615`
- final outside: 27
- final occupancy L1: 27
- final slot p95: `0.1146610826253891`

### C1

- stable PASS: true
- first stable: step 4992
- epsilon switch: 4736
- best step: 6208
- best slot p95: `0.004098494071513414`
- final outside: 0
- final occupancy L1: 0
- final slot p95: `0.005874851252883673`

Winner candidate: `C1`

Recorded causal interpretation:

`R2_NO_STABLE_PASS_WITHIN_BUDGET__C1_XATTN_STABLE_PASS__XATTN_RESCUES_THIS_FROZEN_TRAINING_PROTOCOL__NOT_THEORETICAL_NECESSITY`

Because C1 passed, **C2 did not run**. Any prior continuity text describing C2 as experimentally tested by this run is revoked.

## AR-01 closed result

**Contract:** `41055bd073538d6b`  
**Closed UTC:** `2026-09-07T06:00:52.301304+00:00`  
**Authority:** `canonical/AR01_RESULT_20260907.md`  
**Verdict:** `AR01_NO_TERMINAL_CLOSURE`

AR0:
- first structural PASS step 2368;
- max structural streak 61;
- final check PASS but terminal streak 1;
- terminal stability FAIL;
- final outside 0, parent 1.0, slot p95 `0.0008499497780576348`.

AR1:
- no free-running structural PASS;
- terminal stability FAIL;
- final outside 28, parent 0.50, slot p95 `0.47583022713661194`;
- final teacher-forced structural PASS true, parent 1.0, slot p95 `0.00206305761821568`;
- teacher/free max-abs gap `0.4984188377857208`.

Interpretation: exact minimal residual feedback implementation is not promotable; AR0 residual stability and AR1 exposure/recovery remain separate open questions. No full C3/C4 verdict.

## RigAnything mechanism execution map

Binding reconstruction:

`R2 RUN FAIL -> C1 RUN PASS -> C2 NOT RUN -> C3/C4 SOURCE-ONLY -> AR-01 separate minimal-feedback RUN, both terminal FAIL`

See `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md` for source commit/blob and rationale.

## Detached-artifact authority correction

A detached generated prereg draft described conditional diffusion as shared AR-01 mechanism. It is non-authoritative. Repo prereg SHA `78981b0215c88462d7a2be607274f638494dc1517f88ed882084a4e83d11fae1` freezes the direct three-mode locus head used by the executed notebook.

## Binding anti-conflation rules

1. Source exists != mechanism tested.
2. Preregistered rung != executed rung.
3. Mechanism tested != full formulation tested.
4. FIT1 success != generalization evidence.
5. Green source/contract CI != science PASS.
6. Apparatus failure != model-science FAIL.
7. No gate widening after results.
8. Historical numerical claims with index gaps require exact artifact verification.
9. Previous streak != terminal stability if terminal contiguous streak is required.
10. `AR01_NO_TERMINAL_CLOSURE` cannot be relabeled `AR1 harms` because AR0 also failed the primary terminal gate.
11. Detached drafts cannot override hashed repo prereg/result authority.

## Closing / promotion transaction

An experiment changes continuation authority only after one reconciliation updates:

1. exact prereg/result/provenance pointer;
2. implementation commit/source;
3. `canonical/AUTHORITY_MAP_V1.json`;
4. this ledger;
5. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` if architecture belief changed;
6. `canonical/EXPERIMENT_REGISTRY_V1.json`;
7. `canonical/CONTEXT_STATE_V1.json`;
8. `CURRENT_STATE.md` if stop/go changed;
9. `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`;
10. explicit supersession/revocation where required.

Geppetto promotion remains blocked until scientifically necessary post-AR01 gate(s) close.