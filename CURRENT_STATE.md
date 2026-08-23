# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `IRIS_V2_EXECUTABLE_PREFLIGHT_PASS__EXACT_REPRESENTATION_STUDY_NEXT__TRAINING_FORBIDDEN`

## Read this first

This file is the single continuation authority during the audit. The preserved pre-audit branch is `g0-g1/single-pose-geometry`.

## Current authorization

**NO optimizer run is authorized yet.** CAL/DEV/EXTERNAL remain sealed for learner/model-selection use.

The immediate blocker is no longer a vague “finish all post-corpus work” requirement. Chronological authority is:

1. Stage-A/B audits localized the corpus problems.
2. Stage-B3 found a bounded 62-asset `.blend` authority repair set rather than a systemic 3993-asset failure.
3. Stage-B4 quarantined 6 active non-Basis shape-key assets and authorized read-only repair for the remaining 56.
4. Stage-B5 dry-run passed 56/56 for IRIS/Geppetto geometry; one Arachne capability drop is bounded.
5. Stage-B6 froze SHA-verified repaired geometry without mutating canonical assets/renders.
6. `IRIS_USABLE_CONTROLLED_CORPUS_FREEZE_V1.md` deliberately deferred Stage-B7 publication/rerender and froze the immediate controlled subset at **3930** assets:
   - 56 repair-pending current-A-render mismatches excluded;
   - 6 active shape-key assets excluded;
   - 1 all-eight-view blank asset excluded.
7. That later freeze explicitly returns the critical path to **exact representation/information ceiling → final controlled frontend → controlled learner**, with appearance/domain robustness after controlled success.

Therefore:

- Gate 1 is closed for the controlled path by quarantine/exclusion.
- Gate 2 is bounded for the controlled path by excluding the unpublished repair/quarantine set; it remains a production-corpus publication task, not a reason to block the 3930 controlled experiment.
- Gate 3 appearance recovery remains mandatory before calling IRIS product/natural-domain qualified, but it does **not** block a preregistered geometry-control learner experiment on A.
- The **immediate no-optimizer scientific gate is the frozen `REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md`** on legal observable geometry.
- SOI-2 / product-relevant counterfactual information-limit work is conditional: required if the exact legal representation remains insufficient and always required before an impossibility claim; it is not automatically required before every controlled learner run if the exact representation ceiling passes.

## Controlled corpus authority

Current immediate controlled population: **3930**.

Frozen split counts:

- FIT: 2935 — OPEN
- TUNE: 313 — OPEN
- CAL: 246 — SEALED
- DEV: 270 — SEALED
- EXTERNAL_HOLDOUT: 166 — SEALED

Open FIT+TUNE population: **3248**. Sealed population: **682**.

Source/provider imbalance remains a required reporting dimension; the selected master corpus is dominated by Objaverse and no mean-only result may hide provider/family tails.

## Canonical problem boundary

Shipping input remains `ONE neutral pose x 8 ordered views`.

IRIS owns observable geometric evidence sufficient for a deterministic `SurfaceBuilder`; it does not own hidden mechanical owner IDs, source-rig exact partition, skeleton hierarchy, skinning or mandatory GFDR.

Current legal evidence family:

- P: common/object-frame surface position evidence;
- N: local orientation/normal evidence;
- U_geo: geometric predictive risk only;
- coarse persistence capability;
- fine **local** correspondence evidence;
- direct visibility/alpha and derived support;
- reciprocal/cycle support evidence;
- provenance;
- set-valued ambiguity when singleton evidence is insufficient.

Match ambiguity/confidence is not silently aliased to `U_geo`.

## Why the previous M256 result cannot authorize scale-up

The completed M256 run remains useful diagnostic learner evidence. It showed strong P/N learning and non-random correspondence learning, but its consumer/evaluator did not match the intended final architecture:

- native authority was 1024 while the matcher candidate lattice was fixed 128x128;
- normalized-grid tolerances were used where native-pixel/exact-cell quantities were required;
- `Z_fine` entered global rank fusion despite the later D2 verdict: global role falsified, local retained-top-k role supported;
- ambiguity/singleton policy was not calibrated;
- checkpoint selection was not the final frozen observable panel.

Therefore M256 is **not** an information-limit result and is **not** V2 qualification.

## Research lineage now reconciled

Primary interpretation/contract authorities:

- `canonical/PRODUCT_CONTRACT_V1.md`
- `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`
- `experiments/g0_g1_single_pose_geometry/FRONTEND_NATIVE_1024_CORPUS_CONTRACT_V1.md`
- `experiments/post_corpus_audit/POST_CORPUS_80_CHECK_CLOSURE_MATRIX_V1.md`
- `experiments/post_corpus_audit/POST_CORPUS_STAGE_A_CLOSURE_DECISION_V2.md`
- `experiments/post_corpus_audit/POST_CORPUS_STAGE_B_INTERPRETATION_V1.md`
- `experiments/post_corpus_audit/POST_CORPUS_STAGE_B3_INTERPRETATION_V1.md`
- `experiments/post_corpus_audit/POST_CORPUS_STAGE_B4_INTERPRETATION_V1.md`
- `experiments/post_corpus_audit/IRIS_USABLE_CONTROLLED_CORPUS_FREEZE_V1.md`
- `experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md`
- `experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`
- `audit/IRIS_RESEARCH_TO_EXECUTABLE_ROLE_MATRIX_20260824.md`

Historical `experiments/iris_controlled_v1/` remains provenance/diagnostic only on this branch.

## Active V2 implementation

Active candidate code:

`experiments/iris_single_pose_v2/`

Architecture:

```text
native/control RGBA x 8
      |
shared high-resolution encoder
      |
full local f16 axial reasoning
      |
fixed pooled global multiview context + known yaw
      |
fuse global context back into local field
      |
Z_coarse @ R/8 -------- global high-recall basin admission
      |                         |
P/N/U_geo @ R/2 ---- P rescue  | retain top-k basins
Z_fine @ R/2 ------------------+--> LOCAL refinement only
                                      |
                              reciprocal/cycle SUPPORT
                                      |
                              set-valued hypotheses
                                      |
                            deterministic SurfaceBuilder
```

No fixed `max_w`; no fixed 128 matcher lattice; no global `Z_fine` admission; no premature singleton.

## V2 code corrections closed in this audit

The active path now explicitly fixes several silent mismatches discovered during audit:

1. reciprocal training is same-locus/set-valued aware rather than diagonal-ID-only;
2. accepted correspondence truth retains the exact continuous projected coordinate; raster authority is a visibility/surface-consistency witness and no longer re-quantizes truth to a pixel center;
3. observation-level SupCon subsampling no longer truncates a view-major prefix and bias early views;
4. stage reuse is source-byte fingerprinted;
5. truth-cache reuse is stage/settings/**builder semantic** fingerprinted, including `geometry.py`, so stale truth cannot survive a geometry-semantic change;
6. deterministic reciprocal/cycle is implemented as support evidence and cannot silently delete candidates or authorize a singleton;
7. evaluator uses exact coarse cells, object-space geometry tails and native-1024 pixel localization, with provider/style stratification;
8. evaluator deliberately has **no checkpoint-selection key yet**. Loss-scalar selection is forbidden until the post-representation mini prereg freezes the observable panel.

## Exact committed-byte CI

Draft PR #4 triggers `.github/workflows/iris_v2_preflight.yml`.

Latest verified run: GitHub Actions run `32669024486`, conclusion **SUCCESS**.

Passing exact-commit steps:

- compile all V2 Python sources;
- architecture / coordinate / matcher preflight;
- physical firewall / truth-cache semantic-invalidation preflight;
- observable evaluator synthetic preflight.

This is executable/apparatus evidence only. It is not learner evidence.

## Current next scientific executable

**Do not train.** Execute the frozen exact Representation Authority Study on the controlled/open legal observation surface.

Authority:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md`

Required arms include:

- R0 exact P;
- R1 exact P+N;
- R2 preregistered P-noise ladder;
- R3 preregistered P+N-noise ladder;
- R4 uncertainty-aware relational address only as a bounded ablation, with hard uncertain anchors forbidden.

Required reporting includes top1/top4/top8 containment, canonical localization, reciprocal/cycle, family p90/p95, provider/source and coverage conditioning, and set-valued ambiguity treatment.

### Decision after that study

- If exact P/P+N remains near-ceiling: freeze which legal representation V2 must learn, then run production-width no-optimizer GPU preflight, freeze mini membership + observable checkpoint key + prereg, and only then create one training runner/notebook.
- If exact P/P+N is insufficient but uncertainty-aware relational address materially closes a reproducible hard tail: revise the V2 target/consumer contract **before** training.
- If legal representation remains insufficient: proceed to SOI-2/target-authority investigation; do not blame a learner that has not run.

## Downstream closure remains open

A successful controlled IRIS frontend is not product closure. E0–E5 remain mandatory before claiming equivalent-substrate/product success:

`exact observable substrate -> Geppetto ceiling -> extractability -> predicted-vs-exact consequence -> ambiguity stress -> functionally equivalent 2.5D rigging substrate`.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
