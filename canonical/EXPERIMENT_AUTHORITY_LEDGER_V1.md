# RealSaS — Experiment Authority Ledger V1

**Date:** 2026-09-07  
**Continuation authority:** `CURRENT_STATE.md` on `main`  
**Machine experiment register:** `canonical/AUTHORITY_MAP_V1.json`

This ledger records what an experiment actually tested and what it did **not** test. The live branch/head/status view is generated in `canonical/LIVE_AUTHORITY_MAP.md`; do not copy live branch SHAs here.

## Status vocabulary

- `ACTIVE_RUNNING` — preregistered experiment has not reached its final fixed verdict.
- `CLOSED_PASS` / `CLOSED_FAIL` — exact preregistered gate closed.
- `COMPONENT_EVIDENCE` — mechanism-level evidence only.
- `IMPLEMENTED_NOT_AUTHORITATIVELY_RUN` — executable source exists but does not constitute a scientific result.
- `OPEN_NOT_RUN` — recognized gate with no closed run.
- `INDEX_GAP_NEEDS_BACKFILL` — historical evidence exists, but exact canonical result pointer must be reverified before numerical claims are relied on.

## Current gate matrix

| Gate / experiment | Status | Exact scientific question | What it can establish | What it cannot establish |
|---|---|---|---|---|
| `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE` | **ACTIVE_RUNNING** | Matched AR0 vs AR1: does preregistered joint+parent mechanical recurrent feedback improve Mage FIT1 reachability/retention and parent/root structural prediction? | Verdict on the exact minimal AR-01 feedback intervention under its fixed design | Full RigAnything-formulation verdict; generalization; mechanical salience; automatic product promotion |
| `RIGANYTHING_FULL_FORMULATION_CHALLENGER_C3_C4` | **IMPLEMENTED_NOT_AUTHORITATIVELY_RUN** | Full-surface access + conditional diffusion locus + generated/teacher joint/parent feedback token, with sibling/BFS-order helper available | Existing implementation/design lineage for a fuller formulation-level challenger | Source existence is not evidence; C1/C2/AR-01 component results do not sum into a C3/C4 verdict |
| `GEPPETTO_C1_FULL_SURFACE_ACCESS` | **COMPONENT_EVIDENCE** | Isolate per-step full-surface evidence access | Evidence-access mechanism behavior | Skeleton-causal necessity, diffusion necessity, or full RigAnything equivalence |
| `GEPPETTO_C2_CONDITIONAL_DIFFUSION_LOCUS` | **COMPONENT_EVIDENCE** | Add conditional diffusion locus generation on the matched ladder | Locus-generation mechanism behavior | Joint/parent recurrent feedback or sibling-order effects |
| `V3P_LOSSLESS_EVIDENCE_STABILITY` | **COMPONENT_EVIDENCE** | Preserve lossless fixed 8-view evidence side-path under historical compatibility | Lossless consumer-seam behavior and same-witness stability diagnostics | Generic Geppetto promotion or generalization |
| `V3X_FULL_SURFACE_STABILITY` | **COMPONENT_EVIDENCE** | V3P + per-step full-surface access | Reachability/stability diagnostic for full-surface access | Skeleton-causal AR or formulation closure |
| `FIT1_TRUNK_ACTUAL_UPDATE_CAP_0_005` | **COMPONENT_EVIDENCE** | FIT1-specific optimizer-time strong-trunk update containment | Mage FIT1 containment evidence under preregistered use | Generic architecture stability, generalization, or product-optimal threshold |
| `MAGE_FIT1_HISTORICAL_B1S` | **INDEX_GAP_NEEDS_BACKFILL** | Historical same-witness fit of latent-only Geppetto family | Same-witness capacity once exact result authority is reverified | Structural-AR necessity or generalization |
| `R6_COVERAGE_STRESS` | **INDEX_GAP_NEEDS_BACKFILL** | Coverage stress under preregistered coverage levels | Coverage evidence once corrected-run pointer is normalized | FIT1, AR-01, or generalization closure |
| `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` | **OPEN_NOT_RUN** | Learn whether deform controls are mechanically necessary/simplifiable without functional loss | Future neural responsibility closure | Compiler legality cannot substitute for semantic/mechanical evidence |

## AR-01 scope lock

AR-01 is deliberately **not** a full RigAnything replication. It keeps the lossless seam, full-surface access policy, locus head, serialization, optimizer, seed/backend, count/evaluation policy matched and changes only the preregistered safe mechanical feedback gate.

Therefore:

- `AR1 PASS / AR0 FAIL` means the minimal feedback intervention rescued this FIT1 setting.
- `both PASS` means that intervention is compatible but not necessary on this witness.
- `AR0 PASS / AR1 FAIL` means the intervention harms this FIT1 setting.
- `both FAIL` means AR-01 did not close the problem.

None of those labels is automatically a verdict on the fuller C3/C4 RigAnything-mechanism formulation.

## Binding anti-conflation rules

1. Source exists != mechanism tested.
2. Mechanism tested != full formulation tested.
3. FIT1 witness success != generalization evidence.
4. A green source/contract workflow != scientific PASS.
5. An apparatus/config failure != model-science FAIL.
6. No gate meaning may be widened after results are visible.
7. Numerical historical claims marked `INDEX_GAP_NEEDS_BACKFILL` require exact repo artifact verification before reuse.

## Closing / promotion transaction

An experiment changes continuation authority only after one atomic reconciliation updates:

1. exact prereg/result/provenance pointer;
2. exact implementation commit/source;
3. `canonical/AUTHORITY_MAP_V1.json` active/closed state;
4. this ledger;
5. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` if architecture belief changed;
6. `CURRENT_STATE.md` if continuation/stop-go changed;
7. explicit supersession/revocation of the previous interpretation where applicable.

Until that transaction closes, the live map may show a branch as active, but it is never canonical continuation authority.
