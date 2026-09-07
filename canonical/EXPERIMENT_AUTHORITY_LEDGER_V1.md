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
- `IMPLEMENTED_NOT_AUTHORITATIVELY_RUN` — executable source exists but does not constitute a scientific result.
- `OPEN_NOT_RUN` — recognized gate/question with no closed run.
- `INDEX_GAP_NEEDS_BACKFILL` — historical evidence exists, but exact canonical result pointer must be reverified before numerical claims are relied on.

## Current gate matrix

| Gate / experiment | Status | Exact scientific question | What it can establish | What it cannot establish |
|---|---|---|---|---|
| `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE` | **CLOSED_FAIL — `AR01_NO_TERMINAL_CLOSURE`** | Matched AR0 vs AR1: does preregistered minimal joint+parent mechanical recurrent feedback improve Mage FIT1 reachability/retention and parent/root structural prediction? | Neither arm closed the preregistered terminal 48-check stability gate. AR0 showed strong reachability/long exact retention but residual late collapse; AR1 showed severe teacher-forced/free-running exposure failure. | Full RigAnything-formulation verdict; all structural AR is harmful; scheduled sampling/soft feedback is required; generalization; mechanical salience; automatic promotion |
| `POST_AR01_DIAGNOSIS_NEXT_GATE_SELECTION` | **OPEN_NOT_RUN** | Which remaining uncertainty should be isolated next: AR0 residual terminal stability, AR1 generated-state exposure/recovery, or fuller C3/C4 formulation behavior? | A future prereg can isolate one of these questions. The decision record itself is not scientific evidence. | No mechanism is validated merely by being selected for the next test. |
| `RIGANYTHING_FULL_FORMULATION_CHALLENGER_C3_C4` | **IMPLEMENTED_NOT_AUTHORITATIVELY_RUN** | Full-surface access + conditional diffusion locus + generated/teacher joint/parent feedback token, with sibling/BFS-order helper available | Existing implementation/design lineage for a fuller formulation-level challenger | Source existence is not evidence; C1/C2/AR-01 component results do not sum into a C3/C4 verdict |
| `GEPPETTO_C1_FULL_SURFACE_ACCESS` | **COMPONENT_EVIDENCE** | Isolate per-step full-surface evidence access | Evidence-access mechanism behavior | Skeleton-causal necessity, diffusion necessity, or full RigAnything equivalence |
| `GEPPETTO_C2_CONDITIONAL_DIFFUSION_LOCUS` | **COMPONENT_EVIDENCE / POINTER BACKFILL IN PROGRESS** | Add conditional diffusion locus generation on the matched ladder, with geometry feedback absent | Locus-generation mechanism behavior under its own component protocol | Joint/parent recurrent feedback or sibling-order effects; full C3/C4 formulation verdict |
| `V3P_LOSSLESS_EVIDENCE_STABILITY` | **COMPONENT_EVIDENCE** | Preserve lossless fixed 8-view evidence side-path under historical compatibility | Lossless consumer-seam behavior and same-witness stability diagnostics | Generic Geppetto promotion or generalization |
| `V3X_FULL_SURFACE_STABILITY` | **COMPONENT_EVIDENCE** | V3P + per-step full-surface access | Reachability/stability diagnostic for full-surface access | Skeleton-causal AR or formulation closure |
| `FIT1_TRUNK_ACTUAL_UPDATE_CAP_0_005` | **COMPONENT_EVIDENCE** | FIT1-specific optimizer-time strong-trunk update containment | Mage FIT1 containment evidence under preregistered use | Generic architecture stability, generalization, or product-optimal threshold |
| `MAGE_FIT1_HISTORICAL_B1S` | **INDEX_GAP_NEEDS_BACKFILL** | Historical same-witness fit of latent-only Geppetto family | Same-witness capacity once exact result authority is reverified | Structural-AR necessity or generalization |
| `R6_COVERAGE_STRESS` | **INDEX_GAP_NEEDS_BACKFILL** | Coverage stress under preregistered coverage levels | Coverage evidence once corrected-run pointer is normalized | FIT1, AR-01, or generalization closure |
| `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` | **OPEN_NOT_RUN** | Learn whether deform controls are mechanically necessary/simplifiable without functional loss | Future neural responsibility closure | Compiler legality cannot substitute for semantic/mechanical evidence |

## AR-01 closed result

**Contract:** `41055bd073538d6b`  
**Closed UTC:** `2026-09-07T06:00:52.301304+00:00`  
**Authority:** `canonical/AR01_RESULT_20260907.md`  
**Verdict:** `AR01_NO_TERMINAL_CLOSURE`

The preregistered terminal gate required the final step to be structural PASS and the **final contiguous** PASS streak to be at least 48 checks. An earlier qualifying streak could not remain latched after later failure.

### AR0

- first structural PASS: step 2368;
- first 48-check streak: step 11840;
- maximum observed structural streak: 61 checks;
- final step 16384: structural PASS;
- final contiguous streak: 1;
- terminal stability: FAIL;
- final parent accuracy 1.0, outside 0, slot p95 `0.0008499497780576348`.

AR0 therefore establishes strong same-witness reachability/capacity but not terminal stability.

### AR1

- no free-running structural PASS at any check;
- final outside 28;
- final parent accuracy 0.50;
- final slot p95 `0.47583022713661194`;
- terminal stability: FAIL.

Yet the final teacher-forced model has outside 0, parent accuracy 1.0, structural PASS true and slot p95 `0.00206305761821568`. Teacher-forced vs free-running position max-abs gap is `0.4984188377857208`.

This is strong diagnostic evidence for a teacher-forced/generated-state exposure and recurrent error-amplification problem in the **exact AR-01 minimal feedback implementation**.

## AR-01 scope lock — retained after closure

AR-01 was deliberately **not** a full RigAnything replication. It kept the lossless seam, full-surface access policy, direct three-mode locus head, serialization, optimizer, seed/backend, count/evaluation policy matched and changed only the preregistered mechanical feedback gate.

Therefore the closed result means:

- neither matched arm achieved terminal closure;
- there is no preregistered causal winner;
- the exact minimal residual feedback intervention is not promotable;
- AR0 residual stability remains open;
- AR1 generated-state exposure/recovery remains open.

It does **not** mean:

- C3/C4 failed;
- diffusion failed;
- sibling/BFS handling failed;
- all skeleton-causal AR failed;
- teacher forcing must be replaced by one particular technique.

Read `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md` before proposing another RigAnything-equivalent challenger.

## Detached-artifact authority correction

A generated prereg draft outside repository authority described conditional diffusion as shared AR-01 mechanism. That draft is **not** authoritative. The repository prereg SHA-256 `78981b0215c88462d7a2be607274f638494dc1517f88ed882084a4e83d11fae1` freezes the direct three-mode continuous locus head, matching the executed notebook.

Rule: when a detached/generated artifact conflicts with an immutable repo prereg/result hash, the repo authority wins and the discrepancy must be logged as a continuity hazard.

## Binding anti-conflation rules

1. Source exists != mechanism tested.
2. Mechanism tested != full formulation tested.
3. FIT1 witness success != generalization evidence.
4. A green source/contract workflow != scientific PASS.
5. An apparatus/config failure != model-science FAIL.
6. No gate meaning may be widened after results are visible.
7. Numerical historical claims marked `INDEX_GAP_NEEDS_BACKFILL` require exact repo artifact verification before reuse.
8. A previous stability streak != terminal stability if the prereg requires a terminal contiguous streak.
9. `AR01_NO_TERMINAL_CLOSURE` cannot be relabeled `AR1 harms` merely because AR1 performed worse; AR0 also failed the primary gate.
10. Detached chat/generated drafts cannot override hashed repository preregistration.

## Closing / promotion transaction

An experiment changes continuation authority only after one atomic reconciliation updates:

1. exact prereg/result/provenance pointer;
2. exact implementation commit/source;
3. `canonical/AUTHORITY_MAP_V1.json` active/closed state;
4. this ledger;
5. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` if architecture belief changed;
6. `canonical/EXPERIMENT_REGISTRY_V1.json` dependency/outcome state;
7. `canonical/CONTEXT_STATE_V1.json` fast rehydration state;
8. `CURRENT_STATE.md` if continuation/stop-go changed;
9. `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` append-only claim history;
10. explicit supersession/revocation of the previous interpretation where applicable.

AR-01 is closed, but **Geppetto promotion remains blocked** until the scientifically necessary post-AR-01 gate(s) and a separate refreeze transaction close.