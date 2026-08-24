# P Formulation V4 — CI269 Result Interpretation

Date: 2026-08-24
Status: `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL__NATIVE1024_SCALE_CLOSED__RESIZED_ALPHA_REESTIMATION_FALSIFIED`

## Frozen run authority

- source head: `1c170d6e077dae52e3a6dc171d0d8cabd8eaf528`
- GitHub Actions run: #269 / ID `32752782912`
- artifact ID: `9529705348`
- bundle SHA-256: `a206f5d0015c4d9a66e334f21f3eb8beca62dea169f1b35b13763f6b1ad9a0cf`
- optimizer steps: `0`
- TUNE consumed: `false`
- sealed splits opened: `false`
- model camera half-extent input: `false`

Persistent Drive result:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_CI269_RESULT/P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE.json`

Canonical result label from the frozen V4 preregistration:

`P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`

Fatal assets: `3/16`.

## What passed

The P-V3/P-V4 geometric core remains healthy:

- canonical gauge passed;
- teacher raster -> canonical P projection remained near numerical zero;
- exact-depth analytic P reconstruction remained healthy;
- no camera `half_extent` was passed to the model;
- native image-derived scale is sufficient on the entire 16-asset sentinel panel.

Most importantly, when `h_sheet` is estimated from the **native 1024 RGBA before any resize**, all 16 assets and both styles pass the existing P p95 <= 0.005 gate.

Across the 32 native1024 asset x style cells:

- maximum P p95: `0.0013928374974057078`
- median P p95: approximately `7.32e-05`

Therefore the image-derived canonical scale itself is not falsified at native product resolution.

## What failed

V4 additionally required re-estimating `h_sheet` independently after resize at 512 and again after the exact staged 512->256 transform. That requirement is falsified.

Three assets were fatal:

1. `asset_5890cf3d012a4370ad656761`
   - camera half extent: `0.54`
   - R512: PASS, P p95 ~`7.26e-05`
   - R1024: PASS, P p95 ~`7.26e-05`
   - R256: FAIL, P p95 `0.02480961699038744`

2. `asset_00aa1b666ba193851a498194`
   - camera half extent: `0.5570941257476807`
   - R1024 estimated h: `0.5589519650655022`, P p95 `0.0013928374974057078` — PASS
   - R512/R256 estimated h: `0.7231638418079096`, P p95 `0.12448619268834585` — FAIL

3. `asset_76313e4bd82b82fcd1659c70`
   - camera half extent: `0.6172158837318421`
   - R1024 P p95 `0.0008157795993611216` — PASS
   - R512 P p95 `0.0018853721325285731` — PASS
   - R256 P p95 `0.0051257542567327615` — narrowly FAIL

## Exact resize diagnosis on `asset_00aa...`

The `cel_clean_512.png` derivative is **not a bad render**.

Direct byte-domain image analysis showed that resizing native `cel_clean.png` from 1024 to 512 with the corpus LANCZOS transform reproduces `cel_clean_512.png` pixel-for-pixel:

- RGB MAE = `0`
- alpha MAE = `0`
- alpha-support IoU = `1.0`

The failure is instead caused by the V4 support definition `alpha >= 0.5` after antialiased downsampling.

For the same V0 image:

- native1024 alpha bbox span at threshold 0.5: `916 x 710`
- R512 RGB/support extent remains approximately half-scale (`~463 x 360`)
- R512 alpha bbox at threshold 0.5 contracts to only `324 x 354`

The thin spear/appendage remains visibly represented in resampled RGBA but much of its antialiased alpha falls below 0.5. Thus a hard 0.5 support threshold is not resolution-stable.

This is an observable-preprocessing formulation failure, not a corpus render corruption and not a P geometry failure.

## Canonical interpretation

Do **not** lower the P threshold and do **not** tune a new alpha threshold post-result.

The scientifically justified next formulation is:

1. receive the original/native ordered RGBA sheet;
2. estimate canonical `h_sheet` **once, before spatial downsampling**;
3. carry that deterministic image-derived scalar through any subsequent 512/256 learner preprocessing;
4. reconstruct P at every learner resolution using the same native-derived gauge;
5. learn only camera-forward depth.

In shorthand:

`native RGBA -> observable h_sheet -> resize/features -> depth -> analytic P`

not:

`native RGBA -> resize -> re-estimate h_sheet at each resolution`.

This is consistent with the product input contract and removes the resolution-dependent support-threshold artifact exposed by CI269.

## Next gate

Training remains forbidden.

Before any optimizer step, freeze and run a new optimizer-zero closure that keeps the CI269 panel and P <= 0.005 gate but computes `h_sheet` once from native1024 RGBA and reuses that exact scalar for 1024/512/256 reconstruction.

No TUNE, CAL, DEV, or EXTERNAL split may be opened for this formulation repair.
