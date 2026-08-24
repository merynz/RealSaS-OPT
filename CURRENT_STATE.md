# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__DEPTH_OVERFIT_PREREG_FROZEN__CPU_PREFLIGHT_PASS__GPU_RUN_NEXT`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Current P authority:

- `P_FORMULATION_V4_CI269_RESULT_INTERPRETATION_20260824.md`
- `P_FORMULATION_V5_NATIVE_SCALE_ONCE_CLOSURE_PREREG_20260824.md`
- `P_FORMULATION_V5_CI283_RESULT_INTERPRETATION_20260824.md`
- `model_pv5.py`
- `p_formulation_v5_preflight.py`
- `p_formulation_v5_corpus_audit.py`
- frozen 16-asset FIT panel: `P_FORMULATION_V3_PANEL_V1.json`
- `P_V5_DEPTH_OVERFIT_PREREG_20260824.md`
- `P_V5_DEPTH_OVERFIT_MEMBERSHIP_V1.json`
- `P_V5_DEPTH_OVERFIT_RELEASE_V1.json`

No general learner training is authorized. Only the preregistered tiny FIT-only P-V5 depth overfit diagnostic is authorized next.

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

## P-V4 / CI269 — native observable scale survives; post-resize re-estimation fails

V4 estimated `h_sheet` from ordered RGBA alpha occupancy and kept camera `half_extent` out of model input.

G0 CI269: PASS.

- execution source head `1c170d6e077dae52e3a6dc171d0d8cabd8eaf528`;
- Actions run #269 / ID `32752782912`;
- artifact ID `9529705348`;
- ZIP SHA-256 `a206f5d0015c4d9a66e334f21f3eb8beca62dea169f1b35b13763f6b1ad9a0cf`.

G1 result: `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`, fatal `3/16`, optimizer 0, no TUNE/sealed access.

Crucial positive result: native1024 scale passed P p95 <=`0.005` on all 16 assets × both styles. V4 failed because it re-estimated scale after resize. The 512 derivative of `asset_00aa...` was reproduced pixel-exactly from native1024; the failure was `alpha>=0.5` support changing under antialiased resize, not corpus corruption.

Canonical interpretation: `native observable scale survives; resolution-dependent re-estimation is falsified.`

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

Overall extractor remains image-only apart from the required canonical ordered yaw convention. Teacher camera half-extent is not an extractor/model input.

Learner model boundary:

`IRISSinglePoseV2PV5.forward(images, yaw_deg, sheet_half_extent)`

The scalar is legal only when produced by `estimate_native_sheet_half_extent(native_images, yaw_deg)` from original ordered 1024 RGBA. The helper rejects non-1024 input.

## P-V5 G0/G1 — CLOSED/PASS, CI283

Exact execution-source head:

`e0ea5cf3ba8003ce30e10cf051ca42ce4f1d900f`

GitHub Actions `IRIS V2 Preflight`:

- run #283;
- run ID `32755463835`;
- conclusion `SUCCESS`;
- artifact ID `9530710851`;
- bundle SHA-256 `8e8cc6c723c5f28fcf93a324435bdc8f90ab722b32e75c01feb4953ec912ac68`.

Persisted scientific result:

`P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`

Frozen panel/result integrity:

- 16 FIT-only sentinels;
- 2 styles;
- 3 learner-resolution conditions;
- 96/96 P-p95 cells passed unchanged `<=0.005`;
- fatal assets: 0;
- max P p95 `0.0013928374974057078`;
- median P p95 `7.315552629734155e-05`;
- optimizer steps 0;
- TUNE consumed false;
- sealed splits opened false;
- camera half-extent model input false;
- scale re-estimated after resize false.

Canonical interpretation: `P-V5 native-scale-once geometry formulation is CLOSED/PASS.`

## CURRENT GATE — P-V5 FIT-only depth overfit

Preregistration is frozen before optimizer step 1:

`experiments/iris_single_pose_v2/P_V5_DEPTH_OVERFIT_PREREG_20260824.md`

Membership is exactly the pre-result frozen 8 `FIT_TRAIN_SENTINEL` assets from the V3 panel; both styles are used, yielding 16 within-panel cells. No FIT_SELECT/TUNE/CAL/DEV/EXTERNAL asset participates.

Causal intervention:

- train encoder + within/cross-view reasoning + context/decoder + `p_depth_head`;
- freeze N/U/Z heads;
- optimize only FP32 SmoothL1 on camera-forward depth;
- evaluate full reconstructed canonical P;
- canonical yaw comes from V0..V7 ordering;
- `camera.json` is not consumed by the new stage/cache/training path;
- native `h_native` is derived once from original 1024 RGBA and transported unchanged.

Frozen optimizer protocol:

- R256;
- AdamW lr `3e-4`, betas `(0.9,0.95)`, weight decay `0`;
- 64 epochs × 16 asset-style cells = 1024 planned optimizer steps;
- candidate epochs 16/32/48/64;
- select minimum worst-cell P p95;
- PASS iff every 16 cell has P p95 `<=0.005` at selected checkpoint.

Allowed result labels:

- `P_V5_DEPTH_OVERFIT_PASS`;
- `P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT`.

### Release/preflight state

`P_V5_DEPTH_OVERFIT_RELEASE_V1.json`:

- exact CI283 upstream source byte identity PASS for `coords.py`, `model.py`, `model_pv4.py`, `model_pv5.py`, `geometry.py`;
- source compile PASS;
- frozen membership regression PASS;
- metadata/TUNE/sealed firewall PASS;
- native-1024 scale regression PASS;
- non-1024 native-helper rejection PASS;
- P-only synthetic forward/backward PASS;
- nonzero finite depth-head gradient PASS;
- CI283 scale-transport/R256 field-shape regression PASS;
- synthetic 8-asset stage -> cache -> dataset pipeline PASS;
- local scientific optimizer steps `0`.

GPU capacity preflight is intentionally not claimed locally. The canonical notebook runs an R256 CUDA AMP-forward + FP32 P-only backward with **zero optimizer steps** and fails closed before real training if it does not PASS.

### NEXT EXECUTABLE STEP

Use the canonical notebook:

`RealSaS_IRIS_PV5_Depth_Overfit_V1.ipynb`

Runtime: **CUDA GPU REQUIRED** because it contains training. CPU is sufficient only for preflight/regression.

Run All performs, in order:

1. mount Drive and verify master corpus;
2. verify embedded source bundle SHA;
3. rerun CPU contract + CI283 regression;
4. verify persisted CI283 parent authority;
5. run zero-step CUDA R256 GPU preflight;
6. stage/cache exactly 8 FIT assets to local SSD;
7. freeze pre-optimizer authority;
8. train P-only V5 overfit;
9. evaluate all 16 asset-style cells;
10. persist decision + best checkpoint + authority to:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_V5_DEPTH_OVERFIT_V1`

No broader learner training is authorized by preparation alone.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
