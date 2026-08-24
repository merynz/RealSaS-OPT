# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI202_MINI_COMPLETE__LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT__GENERALIZATION_GAP_NOT_PRIMARY__OPTIMIZATION_DIAGNOSTIC_NEXT`

> **Execution-source rule:** CI202 learner result is bound to the tested code-bearing source head `83a856679a3bc4b4c0c894602f55ac1ffd94aa73` / bundle V4. Later interpretation/docs commits do not change that completed-run authority.

## Single continuation authority

Canonical learner interpretation:

`experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_CANONICAL_INTERPRETATION_CI202_20260824.md`

The sole active candidate implementation remains `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

## Representation gate remains CLOSED/PASS

CI104 canonical label remains `P_GEOMETRY_SUFFICIENT`. Exact P on the frozen 256 OPEN panel / 12,288 queries was top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical-error tails zero. CI202 learner failure does **not** reopen R4/SOI-2 or information existence.

## CI202 completed mini authority

Frozen Mini Extractability / Generalization V1 completed successfully at the apparatus/protocol level:

- FIT_TRAIN128 / FIT_SELECT32 / TUNE_FINAL26.
- selected checkpoint epoch16 / optimizer step512.
- TUNE not used for checkpoint selection.
- TUNE staging started only after checkpoint freeze.
- CI104 representation seed not consumed at learner runtime.
- CAL/DEV/EXTERNAL_HOLDOUT unopened.
- N remained local-orientation diagnostic/supervision only; no correspondence or checkpoint authority.
- completed status: `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

Persistent result root:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_MINI_EXTRACTABILITY_CI202_RESULT`

## What CI202 actually says

The model learned substantial signal but missed absolute promotion precision.

Random-init -> selected epoch16 on FIT_SELECT:

- P p95: 3.868741 -> 0.344555 (91.1% reduction).
- Zc Recall@8: 0.358507 -> 0.789931 (+0.431424).
- P-basin top8: 0.345486 -> 0.649306.
- oracle Zf top1 p95: 51.34 -> 35.95 native px.
- end-to-end top8 hit <=16px: 0.165799 -> 0.827257.
- N p95 diagnostic: 147.63 -> 67.85 deg.

Frozen core thresholds nevertheless fail:

- FIT P p95 <=0.005 required; observed 0.344555.
- FIT Zc@8 >=0.90 required; observed 0.789931.
- FIT oracle Zf p95 <=16 px required; observed 35.947 px.

TUNE absolute family/style criteria also fail, but **generalization gap itself is not the primary blocker**:

- P p95 FIT 0.344555 -> TUNE 0.413841; ratio 1.2011 <= frozen max 1.5.
- Zc@8 FIT 0.789931 -> TUNE 0.762019; drop 0.02791 <= frozen max 0.10.
- end-to-end top8 <=16px FIT 0.827257 -> TUNE 0.823718.
- oracle Zf p95 FIT 35.947 -> TUNE 37.754 px.

Aggregate clean-vs-ink style differences are tiny relative to the absolute error; the hard tail is primarily asset/family driven, not a style-domain collapse.

## Optimization-state diagnosis

Do not infer an architecture ceiling from CI202.

Epoch16 was the best frozen candidate and learning was still improving:

checkpoint score 223.90 (e4) -> 202.84 (e8) -> 198.66 (e12) -> 189.63 (e16).

Between epochs12 and16, training P, Zc and Zf objectives continued improving; Zf local top1 also rose. Thus the 16-epoch mini does not show a clean convergence plateau.

At the same time, do **not** authorize a blind larger/full training run. The next scientific question is whether the gap is optimization-budget limited or a current learner objective/architecture ceiling.

## Normal policy remains frozen

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`.

`geom_n` remains legal observation-local orientation supervision and angular diagnostic only. Production correspondence remains Z_coarse + P global admission and Z_fine local refinement. N stays forbidden from correspondence admission/ranking and checkpoint selection.

## NEXT EXECUTABLE STEP

Preregister and run a **FIT-only Optimization Sufficiency Diagnostic**. Do not reuse TUNE for tuning/selection.

Required diagnostic structure:

1. Tiny same-asset overfit arm: can the current architecture/loss drive legal P and local Zf near target on a deliberately small FIT subset?
2. Longer FIT-only learning-curve arm: does the same frozen learner continue materially improving beyond the 16-epoch budget, or plateau far above target?
3. Keep current P/Zc/Zf/N authority roles fixed; no TUNE/CAL/DEV/EXTERNAL opening during diagnostic selection.

The diagnostic should support only causal interpretations such as:

- `OPTIMIZATION_BUDGET_LIMITED`
- `CURRENT_LEARNER_OBJECTIVE_OR_ARCHITECTURE_LIMITED`
- `MIXED_P_LIMIT__CORRESPONDENCE_HEALTHY`

Do not proceed to SurfaceBuilder/Geppetto/Arachne sufficiency or final1024/product claims until learner extraction is characterized.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
