# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__DEPTH_OVERFIT_COMPLETED_INSUFFICIENT__FIELD_REPRESENTATION_CLOSURE_NEXT`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

### Closed authority

- CI104: `P_GEOMETRY_SUFFICIENT` remains CLOSED/PASS.
- CI202 free-XYZ learner is historical and insufficiently precise.
- P-V3: analytic screen-plane + scalar depth survives; fixed `h=0.54` falsified.
- P-V4: native image-derived scale survives; post-resize scale re-estimation falsified.
- P-V5 / CI283: `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`.
  - 16 FIT sentinels × 2 styles × 3 resolution conditions = 96/96 P-p95 cells <= 0.005.
  - max P p95 `0.0013928374974057078`.
  - optimizer 0; TUNE/sealed closed; teacher camera half-extent not a model input.

P-V5 formulation remains authoritative:

```text
native1024 ordered RGBA
  -> estimate h_native once
  -> transport h_native unchanged
  -> canonical yaw from V0..V7 ordering
  -> learn camera-forward scalar depth d
  -> analytic canonical P
```

`camera.json` remains forbidden in the learner/extractor path.

## Completed diagnostic — P-V5 FIT-only depth overfit

The preregistered 8 FIT asset × 2 style run completed cleanly:

- 1024 optimizer steps;
- selected epoch 64;
- global P p95 `0.08759939931333059`;
- worst-cell P p95 `0.1367238707840442`;
- threshold `0.005`;
- result `P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT`;
- TUNE consumed false;
- sealed splits opened false;
- `camera.json` consumed false.

However, **this result must not yet be interpreted as a pure learner/optimizer failure.** The experiment used an R256 input but produced P/depth on an R/2 = 128×128 field, then bilinearly sampled that field at exact surface loci. CI283 closed analytic V5 geometry/scale transport, but did not close this neural output-field discretization.

Therefore the old next-policy text `localize learner/optimizer/feature capacity` is now refined by the research-order rule: close output representation before assigning learner blame.

## CURRENT GATE — P-V5 field representation closure

Authority files:

- `P_V5_FIELD_REPRESENTATION_CLOSURE_PREREG_20260824.md`
- `P_V5_FIELD_REPRESENTATION_MEMBERSHIP_V1.json`
- `pv5_field_representation_closure.py`
- `pv5_field_representation_preflight.py`

Question: can a free scalar depth field itself represent the required P precision on the same frozen legal surface truth?

Frozen candidates for R256 learner input:

- 64×64 = R/4;
- 128×128 = R/2 (current P-V5 learner field);
- 256×256 = R.

Oracle semantics:

- exact same 8 pre-result FIT_TRAIN sentinels;
- 4096 deterministic visible raster-authority samples/view using the same `pv5-depth` seed;
- native image-derived `h_native` once per style;
- no `camera.json`;
- no CNN/model weights;
- no TUNE/CAL/DEV/EXTERNAL;
- neural optimizer steps 0;
- free field solved by deterministic sparse LSMR under exact bilinear/border/align_corners=False sampling semantics;
- full canonical P Euclidean error is authority.

A resolution is **CERTIFIED** only if all 16 asset-style cells have P p95 <= 0.005. A failing resolution is `NOT_CERTIFIED`, **not** a mathematical impossibility claim because L2 oracle minimization is not an exact p95 minimax proof.

### Local preflight evidence

Synthetic field/sampling preflight: PASS.

Frozen real sentinel diagnostic on `asset_36fb02305846592b1ecdf3d4`, V7, same 4096 sample seed:

- 64×64 depth abs p95 `0.03989342867777104`;
- 128×128 depth abs p95 `0.013460239341135558` after 10,000 LSMR iterations;
- 256×256 depth abs p95 `4.561941102654288e-13`, max `0.0039734749531841335`.

This is only a one-view preflight. It is enough to justify the full frozen closure gate, but **not** enough to claim the 8-asset R256 field is closed before the canonical run.

## NEXT EXECUTABLE STEP

Run:

`RealSaS_IRIS_PV5_Field_Representation_Closure_V1.ipynb`

Runtime: **CPU is sufficient. GPU is not required. There is no neural training.**

Run All order:

1. mount Drive;
2. verify previous P-V5 depth-overfit lineage/firewall;
3. verify embedded source SHA;
4. rerun CPU solver/grid-sampling preflight;
5. execute frozen 8-asset / 16-cell field closure at 64/128/256;
6. persist `P_V5_FIELD_REPRESENTATION_CLOSURE.json` and `RUN_COMPLETE_P_V5_FIELD_REPRESENTATION_CLOSURE_V1.json`.

Output root:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_V5_FIELD_REPRESENTATION_CLOSURE_V1`

### Next policy after closure

- If 256 is certified and 128 is not: preregister **R256 one-asset / one-style learner overfit** only.
- If 128 is certified: keep R/2 and localize learner/optimizer with one-cell overfit before changing representation.
- If none is certified: reopen output field representation only; do not reopen P ontology or V5 analytic geometry.

No new neural training is authorized until this CPU-only gate is closed.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
