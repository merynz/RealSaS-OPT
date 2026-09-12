# Full-subject IRIS reclosure preregistration — 2026-09-12

**Status:** `PREREGISTERED__NOT_YET_EXECUTED`

## Question

Can the existing scene-first signed-geometry architecture close Mage FIT when the FIT authority matches the complete rendered product subject instead of the historical body-only/subset teacher geometry?

## Frozen product inputs

Inference input remains exactly:

- 8 × 1024 RGBA observations;
- 8 exact orthographic cameras.

No source mesh, source skin, source object names, component labels or teacher geometry may enter product inference.

## FIT-only teacher authority

Use exact normalized full-source geometry SHA-256:

`528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f`

Teacher geometry for mechanical signed-surface FIT consists of the skin-supported subset:

- 5321 source vertices / 5763 source faces total;
- 42 exact zero-skin source vertices are excluded from mechanical teacher authority;
- resulting mechanical teacher subset: 5279 vertices / 5683 eligible faces.

This teacher is training/evaluation authority only.

## Architecture

Freeze the promoted scene-first signed geometry architecture:

`RealSaS.IRIS.SceneFirstSignedGeometry.v3`

Mechanism:

`all-view/all-patch image evidence + exact camera context -> shared scene memory -> continuous signed field -> zero level surface`

Do not introduce component-name conditioning, source mesh conditioning, teacher mesh inference inputs, categorical Mage labels, or downstream skeleton/skin inputs.

## Warm start policy

The historical signed-geometry checkpoint may be used as initialization because the architecture is retained, but its old Mage FIT promotion is invalidated as product authority. Initialization does not transfer old S lineage or closure status.

Run two arms before any architecture change:

1. `FROZEN_OLD_CHECKPOINT_EVAL_ONLY` — no optimizer; demonstrates the already-measured full-subject failure under corrected evaluation authority.
2. `CORRECTED_FULL_SUBJECT_FIT` — same architecture, corrected full-subject FIT authority, optimizer allowed.

No alternate architecture is authorized unless arm 2 fails preregistered gates.

## Numerical/training policy

Retain historical H1 V2 objective semantics:

- local first-hit signed supervision;
- product-alpha visual-hull outside negatives;
- background positive-margin loss;
- signed zero-level extraction by marching cubes;
- product-input-only visual-hull clipping.

Memory-only batching/chunking changes are allowed when they preserve the same objective, sample schedule, model architecture and final tensors within declared numerical tolerance. Batch-size or sample-distribution changes are not silently allowed.

## Gates

Historical geometry p95 / sign gates remain diagnostic, but the reclosure adds mandatory product-aligned gates that the old promotion failed to protect.

For every one of the eight views report:

- source alpha recall;
- source alpha precision;
- source alpha IoU;
- zero-surface silhouette recall / precision / IoU;
- per-component observed coverage where component truth is available.

Hard reclosure requirements:

- exact observation/camera hashes match authority;
- full-source evaluator replay min-view alpha recall >= 0.98;
- zero-surface min-view silhouette recall >= 0.96;
- zero-surface min-view silhouette IoU >= 0.94;
- no required visible component silently absent;
- signed zero bracket rate >= 0.90;
- bidirectional FIT geometry p95 normalized <= 0.05;
- teacher/source geometry absent from product inference graph;
- no downstream G/S/W/CDT artifact is used as an IRIS input.

The 0.96 recall / 0.94 IoU values reuse the previously frozen prospective geometry thresholds rather than being invented after observing corrected-model results.

## Failure interpretation

- frozen old checkpoint failure does not authorize architecture change;
- corrected full-subject fit PASS -> re-emit dense zero-surface, then GSA;
- corrected full-subject fit FAIL -> stop at IRIS and investigate objective/model capacity before any downstream work;
- no threshold widening is authorized by failure.

## Claim boundary

A PASS closes only Mage full-subject IRIS FIT geometry. It does not reclose GSA, Geppetto, Arachne, CDT/MWB2, appearance, runtime, unseen generalization or PRODUCT_PASS.
