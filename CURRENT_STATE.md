# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI244_V3_FIXED_SCALE_FALSIFIED__P_V4_G0_CI269_PASS__REAL_CORPUS_G1_READY__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Current P formulation authority:

- `P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_PREREG_20260824.md`
- `model_pv4.py`
- `p_formulation_v4_preflight.py`
- `p_formulation_v4_corpus_audit.py`
- frozen 16-asset FIT panel: `P_FORMULATION_V3_PANEL_V1.json`

**TRAINING FORBIDDEN.** No learner optimizer step is authorized by the current gate.

## Representation gate remains CLOSED/PASS

CI104 canonical label remains `P_GEOMETRY_SUFFICIENT`: exact legal P is sufficient for same-locus correspondence on the frozen representation panel. Current work changes the extractor parameterization, not P ontology.

P = canonical/object-frame position of the observed physical surface locus.

N remains observation-local orientation supervision/diagnostic only and is forbidden from correspondence admission/ranking and checkpoint selection.

## Historical learner diagnostic — CI202

The old free 3-channel XYZ P learner completed cleanly but was not precise enough:

- FIT_SELECT P p95 `0.344555` vs frozen `<=0.005`;
- Zc@8 `0.789931`;
- oracle Zf p95 `35.947` native px;
- TUNE P p95 `0.413841`.

This is historical extraction evidence only. The checkpoint is incompatible with P-V3/P-V4.

## P-V3 diagnosis — geometry healthy, fixed acquisition scale falsified

P-V3 changed free XYZ regression to:

```text
P = h*gx*right(yaw)
  - h*gy*up
  + depth*forward(yaw)
```

Only camera-forward depth is learned.

CI244/V2 was the valid optimizer-zero 16-FIT real-corpus geometry closure for the fixed-`h=0.54` version.

CI244 authority:

- source head `985997ba87faa92cc004ad9ab70efeacc400886f`;
- Actions run #244 / ID `32746353085`;
- artifact ID `9527242584`;
- ZIP SHA-256 `e04bc7c9638038688304b558cb816f8bccaf5e253bc2195977c0c2328f693667`.

Result: `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`, fatal `2/16` only.

Across all 16 assets:

- canonical bbox center max ~`2.98e-08`;
- largest bbox extent ~`1.0`;
- max absolute canonical coordinate `0.5`;
- raster/P projection p95 max ~`6.72e-08`;
- exact-depth analytic reconstruction p95 max ~`4.00e-08`.

Two internally healthy assets alone falsified fixed `h=0.54`:

- `asset_00aa1b666ba193851a498194`: camera half extent `0.5570941257476807`;
- `asset_76313e4bd82b82fcd1659c70`: camera half extent `0.6172158837318421`.

Therefore the canonical interpretation is:

`P geometry healthy; hardcoded acquisition scale falsified.`

CI237 prior to this is preserved as an apparatus false reject caused by mistakenly requiring the master NPZ field set to equal `{vertices,faces}`. It is not scientific geometry evidence.

## P-V4 — observable image-derived sheet scale

Product IRIS remains image-only. `camera.json half_extent` is **not** a model input.

Canonical geometry has max bbox extent one. For ordered RGBA views:

```text
w0 = V0 alpha bbox width / R
w2 = V2 alpha bbox width / R
hz = max alpha bbox height over V0..V7 / R
m = max(w0,w2,hz)
h_sheet = 1/(2*m)
```

Then:

```text
P = h_sheet*gx*right(yaw)
  - h_sheet*gy*up
  + depth*forward(yaw)
```

`h_sheet` is deterministic observable gauge from RGBA alpha; it is not a learned latent and does not consume teacher camera scale. The depth head receives `(gx,gy,sin(yaw),cos(yaw),h_sheet)`.

## P-V4 G0 — CLOSED/PASS, CI269

Exact execution-source head:

`1c170d6e077dae52e3a6dc171d0d8cabd8eaf528`

GitHub Actions `IRIS V2 Preflight`:

- run #269;
- run ID `32752782912`;
- conclusion `SUCCESS`.

Immutable artifact:

- name `iris-v2-p-formulation-v4-observable-scale-closure-bundle-v1`;
- artifact ID `9529705348`;
- size `34301` bytes;
- ZIP SHA-256 `a206f5d0015c4d9a66e334f21f3eb8beca62dea169f1b35b13763f6b1ad9a0cf`.

CI269 PASS includes:

- exact committed-source compile;
- all historical architecture/coordinate/matcher/cache/AMP/evaluator regressions;
- master-geometry superset firewall;
- P-V4 image-only forward signature (`images,yaw_deg` only);
- alpha-derived scale fixtures around `0.54`, `0.557`, `0.617`;
- P-V4 model forward at input resolutions 256/512/1024 with P at R/2 and Zc at R/8;
- finite full loss/backward and finite nonzero depth-head gradient;
- P-V4 corpus CLI/panel firewall;
- isolated bundle dependency/content replay;
- uploadable bundle verification.

The downloaded artifact was independently replayed outside the repository checkout. Its ZIP SHA matched the GitHub artifact digest; every internal `SHA256SUMS.txt` entry passed; isolated `py_compile`, P-V4 preflight, master-geometry firewall preflight and corpus CLI import all passed.

CI267 is preserved as an apparatus-only failure: the V4 runner imported the old nonexistent symbol `load_primary_geometry`; the legal helper is `load_legal_primary_geometry`. No scientific G1 metric opened in CI267. CI269 closes this symbol drift.

## Drive execution authority

CI269 artifact mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_BUNDLE_CI269.zip`

Drive file ID:

`1I15HBIzVVOitSKZtX2SR9Jh0gbWowEAr`

Canonical Colab notebook:

`RealSaS_IRIS_P_Formulation_V4_Observable_Scale_Closure_CI269.ipynb`

Notebook SHA-256:

`8809874b243e4fc9bef27bf953152a2afedc218433ad186cae4b1da444147d65`

Persistent result root:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_CI269_RESULT`

## CURRENT GATE — P-V4 G1 real-corpus observable-scale closure

Run the canonical CI269 notebook only.

Frozen population:

- same 16 FIT-only sentinels;
- 8 FIT_SELECT + 8 FIT_TRAIN;
- no substitution;
- no TUNE/CAL/DEV/EXTERNAL.

For each asset × style (`cel_clean`,`ink_cel`) × resolution (`256`,`512`,`1024`):

- derive `h_sheet` only from RGBA alpha;
- use exact teacher depth solely as optimizer-zero formulation ceiling;
- reconstruct full 3D P;
- require P p95 `<=0.005`.

Camera half extent is teacher-side diagnostic/reference only and is never passed to the model.

Allowed G1 labels are exactly:

- `P_V4_OBSERVABLE_SCALE_GEOMETRY_CLOSED`
- `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`

The notebook persists either scientific PASS or scientific FAIL. Apparatus SHA/content/import failures hard-stop separately.

## Next gate policy

Only if P-V4 G1 returns `P_V4_OBSERVABLE_SCALE_GEOMETRY_CLOSED` may a **separate FIT-only depth/P overfit preregistration** be frozen before any optimizer step.

Do not reuse CI202 training authority. Do not open TUNE during formulation/overfit debugging. No sealed split is authorized.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
