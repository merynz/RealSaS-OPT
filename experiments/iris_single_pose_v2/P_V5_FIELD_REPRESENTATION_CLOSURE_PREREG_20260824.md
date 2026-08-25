# RealSaS IRIS Single Pose V2 — P-V5 Field Representation Closure V1

Date: 2026-08-24
Status: FROZEN BEFORE EXECUTION
Parent authority: `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED` (CI283)
Historical learner result: `P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT` (1024 optimizer steps); this result is not used to select assets or oracle targets.

## Question

Before blaming learner capacity/optimizer, can the spatial depth-field representation used by P-V5 itself carry the required canonical P precision on the frozen FIT-only surface loci?

This is an optimizer-zero **neural** gate. There is no CNN, no image learner, no checkpoint selection, and no TUNE/sealed access. A deterministic sparse linear oracle fits free scalar camera-forward depth fields to the same legal surface truth used by the P-V5 depth overfit.

## Frozen membership

Use exactly the eight pre-result `FIT_TRAIN_SENTINEL` assets already frozen for the P-V5 depth-overfit gate. No substitution and no post-result asset selection.

Styles: `cel_clean`, `ink_cel`.

## Legal inputs

- `primary_geometry.npz`
- `raster_authority.npz`
- native ordered 1024 RGBA only for the already-approved V5 image-derived `h_native`
- canonical view index `V0..V7 -> yaw 0,45,...315`

Forbidden:
- `camera.json`
- teacher camera half extent as oracle/model input
- TUNE/CAL/DEV/EXTERNAL
- neural model weights or learned features

## Surface truth and sampling

Reproduce the depth-overfit cache exactly:
- 4096 visible raster-authority rows per view (or all if fewer)
- deterministic seed `sha256(f"{asset_id}|pv5-depth|{view}")`
- exact surface P reconstructed from triangle id + barycentric coordinates
- continuous observation coordinate is the 1024 raster pixel-center grid coordinate

## Candidate output representations

For a learner input resolution R=256, test free scalar depth fields at:

- 64x64 (`R/4`)
- 128x128 (`R/2`, current learner P-V5 field)
- 256x256 (`R`, candidate full-resolution field)

Sampling semantics must exactly match `torch.grid_sample(..., mode='bilinear', padding_mode='border', align_corners=False)`.

## Oracle solver

For each asset/view/resolution independently, construct the sparse bilinear sampling matrix `A` and solve

`min_d ||A d - depth_truth||_2`

with deterministic SciPy `lsmr` at tight tolerances. This is a **sufficiency oracle**, not a p95-optimal impossibility proof. Therefore:

- a resolution that passes certifies expressivity for this frozen panel;
- a resolution that fails is only `NOT_CERTIFIED` by this oracle and must not be called mathematically impossible.

## P reconstruction and authority

For each style, derive `h_native` once from the original 1024 RGBA alpha support using the frozen V5 estimator. Do not re-estimate after resize.

At each exact sampled observation `(gx,gy)`:

`P_pred = h_native*gx*right(yaw) - h_native*gy*up + d_oracle*forward(yaw)`

Authority metric: Euclidean canonical `||P_pred - P_truth||`.

## Gate

Per resolution, PASS iff **every one of the 16 asset-style cells** has `P_p95 <= 0.005` over all eight views / 32768 samples per cell.

Overall labels:
- `P_V5_FIELD_REPRESENTATION_CLOSED` if at least one candidate resolution is certified.
- `P_V5_FIELD_REPRESENTATION_NOT_CLOSED` if none is certified.

The report must record the smallest certified candidate in the frozen order `64 -> 128 -> 256`.

## Next policy

If 256 is certified and 128 is not certified:
- preserve V5 geometry/native-scale formulation;
- authorize preregistration only of an **R256 one-asset / one-style learner overfit**;
- do not jump directly to 8-asset training.

If 128 is certified:
- do not change field resolution yet; localize optimization/learner behavior with one-cell overfit first.

If no candidate is certified:
- reopen output representation only (not P ontology, not V5 analytic geometry).

No product/generalization claim is authorized by this gate.
