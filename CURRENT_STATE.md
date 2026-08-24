# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_NATIVE_SCALE_ONCE_G1_PASS__FORMULATION_CLOSED__OPTIMIZER_ZERO__NEXT_FIT_ONLY_DEPTH_OVERFIT_PREREG`

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

No general learner training is authorized yet. The next optimizer-bearing experiment must first receive a separately frozen FIT-only depth/P overfit preregistration.

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

PASS includes native 1024 scale helper boundary, explicit rejection of 512 input, exact scalar transport through R1024/R512/R256, finite full loss/backward, nonzero depth-head gradient, frozen panel firewall, isolated bundle replay and artifact upload.

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_P_FORMULATION_V5_NATIVE_SCALE_ONCE_CLOSURE_BUNDLE_CI283.zip`

Drive file ID: `12oBvRrcXOYkMWuiSD79nHNt7A3jV8uU2`.

## P-V5 G1 — CLOSED/PASS

Persisted scientific result:

`P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`

Result root:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V5_NATIVE_SCALE_ONCE_CLOSURE_CI283_RESULT`

Result SHA-256:

`e233e5f9dfc15f7888d637e1c047094836f40152b4ab5a066bf49b8569c671c7`

Frozen panel/result integrity:

- 16 FIT-only sentinels;
- 2 styles;
- 3 learner-resolution conditions;
- 96/96 P-p95 cells passed the unchanged `<=0.005` gate;
- fatal assets: 0;
- max P p95 `0.0013928374974057078`;
- median P p95 `7.315552629734155e-05`;
- optimizer steps 0;
- TUNE consumed false;
- sealed splits opened false;
- camera half-extent model input false;
- scale re-estimated after resize false.

Worst hard-scale asset `asset_00aa...` uses native-derived h `0.5589519650655022`; the exact same scalar is transported at R1024/R512/R256 and P p95 is `0.0013928374974057078` in all three conditions.

`asset_76313...` uses native-derived h `0.6183574879227053`; P p95 is `0.0008157795993611216` across all three conditions.

Canonical interpretation:

`P-V5 native-scale-once geometry formulation is CLOSED/PASS.`

No threshold was loosened and no asset-specific rescue branch was introduced.

### Notebook finalization typo

The first CI283 notebook had a post-result Python typo `allowed = {{...}}`, which raised `TypeError: unhashable type: 'set'` after the scientific result JSON had already been written. The scientific runner/result are valid; only completion metadata/authority-copy finalization was interrupted.

Recovery notebook:

`RealSaS_IRIS_P_Formulation_V5_Native_Scale_Once_Closure_CI283_RECOVERY.ipynb`

It detects the existing scientific result and does **not** rerun G1. It revalidates the frozen contract, writes the completion marker and copies authority files.

## CURRENT GATE — freeze FIT-only depth/P overfit diagnostic

P formulation is closed at optimizer zero. The next question is learned depth extractability/optimization only.

Before any optimizer step:

1. freeze a new FIT-only overfit preregistration for P-V5;
2. keep TUNE/CAL/DEV/EXTERNAL closed;
3. use native-derived transported scale exactly as closed above;
4. test whether the one-scalar depth learner can drive reconstructed P to the existing precision target on a tiny FIT subset before reopening broader optimization.

No general mini retraining is authorized until this new overfit gate is frozen.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
