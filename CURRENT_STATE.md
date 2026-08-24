# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI269_V4_FAIL__NATIVE1024_SCALE_CLOSED__P_V5_G0_CI283_PASS__G1_READY__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Current P authority:

- `P_FORMULATION_V4_CI269_RESULT_INTERPRETATION_20260824.md`
- `P_FORMULATION_V5_NATIVE_SCALE_ONCE_CLOSURE_PREREG_20260824.md`
- `model_pv5.py`
- `p_formulation_v5_preflight.py`
- `p_formulation_v5_corpus_audit.py`
- frozen 16-asset FIT panel: `P_FORMULATION_V3_PANEL_V1.json`

**TRAINING FORBIDDEN.** No learner optimizer step is authorized by the current gate.

## Representation gate remains CLOSED/PASS

CI104 canonical label remains `P_GEOMETRY_SUFFICIENT`: exact legal P is sufficient for same-locus correspondence on the frozen representation panel. Current work changes extraction/parameterization, not P ontology.

P = canonical/object-frame position of the observed physical surface locus.

N remains observation-local orientation supervision/diagnostic only and is forbidden from correspondence admission/ranking and checkpoint selection.

## Historical learner — CI202

The old free XYZ P learner completed cleanly but was insufficiently precise:

- FIT_SELECT P p95 `0.344555` vs frozen `<=0.005`;
- Zc@8 `0.789931`;
- oracle Zf p95 `35.947` native px;
- TUNE P p95 `0.413841`.

That checkpoint is historical and incompatible with P-V3/V4/V5.

## P-V3 — analytic screen plane survives; fixed h falsified

P-V3 changed P to:

```text
P = h*gx*right(yaw)
  - h*gy*up
  + depth*forward(yaw)
```

Only depth is learned.

Valid CI244/V2 real-corpus closure showed canonical geometry, raster/P projection and exact-depth reconstruction healthy near numerical precision, while two assets falsified fixed `h=0.54` by using `0.557094...` and `0.617216...`.

Canonical interpretation: `P geometry healthy; hardcoded acquisition scale falsified.`

## P-V4 / CI269 — image-derived scale at native1024 passes; post-resize re-estimation fails

V4 estimated `h_sheet` from ordered RGBA alpha occupancy and kept camera `half_extent` out of model input.

G0 CI269: PASS.

- execution source head `1c170d6e077dae52e3a6dc171d0d8cabd8eaf528`;
- Actions run #269 / ID `32752782912`;
- artifact ID `9529705348`;
- ZIP SHA-256 `a206f5d0015c4d9a66e334f21f3eb8beca62dea169f1b35b13763f6b1ad9a0cf`.

G1 frozen result: `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`, fatal `3/16`, optimizer 0, no TUNE/sealed access.

Crucial positive result: native1024 scale passed P p95 <=`0.005` on **all 16 assets x both styles**.

Across the 32 native1024 asset/style cells:

- max P p95 `0.0013928374974057078`;
- median P p95 ~`7.32e-05`.

V4 failed only because it re-estimated the scale after resize.

`asset_00aa...` diagnosis:

- native1024 h `0.5589519650655022`, P p95 `0.0013928374974057078` PASS;
- R512/R256 re-estimated h `0.7231638418079096`, P p95 `0.12448619268834585` FAIL.

The 512 derivative is **not** corrupt. Native 1024 -> corpus 512 LANCZOS reproduction is pixel-exact (RGB MAE 0, alpha MAE 0, alpha IoU 1). The failure is hard `alpha>=0.5` support after antialiased resize: thin support remains visually represented but falls below the threshold.

Therefore:

`native observable scale survives; resolution-dependent re-estimation is falsified.`

Do not tune alpha threshold post-result.

## P-V5 — native-scale-once formulation

Frozen execution boundary:

```text
original ordered native1024 RGBA
    -> estimate h_native ONCE
    -> detach/freeze scalar
    -> resize learner image as needed
    -> learner predicts depth using transported h_native
    -> analytic full P
```

Forbidden:

`resize -> re-estimate h from degraded alpha`.

Overall extractor remains image-only. Teacher camera half-extent is not an extractor/model input.

Learner model boundary:

`IRISSinglePoseV2PV5.forward(images, yaw_deg, sheet_half_extent)`

The scalar is legal only when produced by `estimate_native_sheet_half_extent(native_images, yaw_deg)` from original ordered 1024 RGBA. The helper rejects non-1024 input.

## P-V5 G0 — CLOSED/PASS, CI283

Exact execution-source head:

`e0ea5cf3ba8003ce30e10cf051ca42ce4f1d900f`

GitHub Actions `IRIS V2 Preflight`:

- run #283;
- run ID `32755463835`;
- conclusion `SUCCESS`.

Immutable artifact:

- `iris-v2-p-formulation-v5-native-scale-once-closure-bundle-v1`;
- artifact ID `9530710851`;
- ZIP SHA-256 `8e8cc6c723c5f28fcf93a324435bdc8f90ab722b32e75c01feb4953ec912ac68`.

PASS includes:

- all historical architecture/cache/matcher/AMP/evaluator/firewall regressions;
- P-V5 native 1024 scale helper boundary;
- explicit rejection of 512 input by the native-scale helper;
- learner forward cannot re-estimate scale;
- no camera metadata/half-extent argument;
- same native-derived scalar transported exactly through learner R1024/R512/R256;
- P fields at R/2 and Zc at R/8;
- finite full loss/backward and finite nonzero depth-head gradient;
- frozen panel firewall;
- isolated bundle dependency/content replay;
- artifact upload.

Independent post-CI artifact replay also passed: ZIP digest matched GitHub, every `SHA256SUMS.txt` entry passed, isolated compile/firewall/P-V5 preflight/corpus CLI import passed.

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_P_FORMULATION_V5_NATIVE_SCALE_ONCE_CLOSURE_BUNDLE_CI283.zip`

Drive file ID:

`12oBvRrcXOYkMWuiSD79nHNt7A3jV8uU2`

Canonical notebook:

`RealSaS_IRIS_P_Formulation_V5_Native_Scale_Once_Closure_CI283.ipynb`

Notebook SHA-256:

`86d64a0d63b78cfff847c19ba04cfcdd7ed0ff7936170a46b9d2cfab71d7cb38`

## CURRENT GATE — P-V5 G1 real-corpus native-scale-once closure

Run only the canonical CI283 notebook.

Frozen population:

- same 16 FIT-only sentinels;
- 8 FIT_SELECT + 8 FIT_TRAIN;
- no substitution;
- no TUNE/CAL/DEV/EXTERNAL.

For each asset/style:

1. derive `h_native` once from native1024 alpha with the unchanged V4 native estimator;
2. carry the exact scalar unchanged to R1024/R512/R256 conditions;
3. do not recompute scale after resize;
4. use exact teacher depth only for optimizer-zero formulation ceiling;
5. require full canonical P p95 <=`0.005` at all three learner-resolution conditions.

Required transport invariant:

`h_used_R1024 == h_used_R512 == h_used_R256` exactly.

Allowed labels only:

- `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`
- `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_FAIL`

Only after G1 PASS may a **separate FIT-only depth/P overfit preregistration** authorize optimizer steps.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
