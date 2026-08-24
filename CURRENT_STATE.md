# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI269_V4_FAIL__NATIVE1024_OBSERVABLE_SCALE_CLOSED__RESIZED_ALPHA_REESTIMATION_FALSIFIED__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Current interpretation authority:

- `P_FORMULATION_V4_CI269_RESULT_INTERPRETATION_20260824.md`
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

CI244/V2 was the valid optimizer-zero 16-FIT real-corpus geometry closure for fixed `h=0.54`.

Result: `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`, fatal `2/16` only.

Across all 16 assets:

- canonical bbox center max ~`2.98e-08`;
- largest bbox extent ~`1.0`;
- max absolute canonical coordinate `0.5`;
- raster/P projection p95 max ~`6.72e-08`;
- exact-depth analytic reconstruction p95 max ~`4.00e-08`.

Two internally healthy assets falsified fixed `h=0.54`:

- `asset_00aa1b666ba193851a498194`: camera half extent `0.5570941257476807`;
- `asset_76313e4bd82b82fcd1659c70`: camera half extent `0.6172158837318421`.

Canonical interpretation: `P geometry healthy; hardcoded acquisition scale falsified.`

CI237 is preserved as an apparatus false reject caused by mistakenly requiring the master NPZ field set to equal `{vertices,faces}`. It is not scientific geometry evidence.

## P-V4 G0 — CLOSED/PASS, CI269

P-V4 kept image-only inference and replaced fixed h with deterministic scale estimated from RGBA alpha occupancy. Camera `half_extent` is not a model input.

Exact CI269 execution-source head:

`1c170d6e077dae52e3a6dc171d0d8cabd8eaf528`

GitHub Actions:

- run #269 / ID `32752782912`;
- conclusion `SUCCESS`;
- artifact `iris-v2-p-formulation-v4-observable-scale-closure-bundle-v1`;
- artifact ID `9529705348`;
- ZIP SHA-256 `a206f5d0015c4d9a66e334f21f3eb8beca62dea169f1b35b13763f6b1ad9a0cf`.

G0 included 256/512/1024 model-forward shape checks, image-only forward signature, observable-scale synthetic fixtures, full finite loss/backward, nonzero depth gradient, cache/matcher/AMP/evaluator regressions, and isolated bundle replay.

## P-V4 G1 — SCIENTIFIC FAIL, but native1024 scale CLOSED

Persistent result root:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_CI269_RESULT`

Canonical frozen label:

`P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`

Run hygiene:

- optimizer steps `0`;
- training authorized `false`;
- TUNE consumed `false`;
- sealed splits opened `false`;
- model camera half-extent input `false`.

Fatal assets: `3/16`.

### Crucial positive result

For **native1024 RGBA**, all 16 assets x both styles pass the existing P p95 <= `0.005` gate.

Across the 32 native1024 asset x style cells:

- max P p95 = `0.0013928374974057078`;
- median P p95 ~`7.32e-05`.

Therefore image-derived canonical scale is sufficient at native product resolution on the frozen sentinel panel.

### What actually failed

V4 re-estimated the scale independently from the resized alpha at 512 and 256. That resolution-dependent re-estimation is falsified.

Fatal cases:

- `asset_5890cf3d012a4370ad656761`: R512/R1024 PASS; R256 P p95 `0.02480961699038744` FAIL.
- `asset_00aa1b666ba193851a498194`: R1024 h=`0.5589519650655022`, P p95 `0.0013928374974057078` PASS; R512/R256 h=`0.7231638418079096`, P p95 `0.12448619268834585` FAIL.
- `asset_76313e4bd82b82fcd1659c70`: R1024 P p95 `0.0008157795993611216` PASS; R512 `0.0018853721325285731` PASS; R256 `0.0051257542567327615` narrowly FAIL.

### `asset_00aa...` resize diagnosis

The 512 derivative is **not** a bad render/corrupt derivative.

Native `cel_clean.png` resized 1024->512 with the corpus LANCZOS operation reproduces `cel_clean_512.png` pixel-for-pixel:

- RGB MAE `0`;
- alpha MAE `0`;
- alpha-support IoU `1.0`.

The failure comes from hard support definition `alpha >= 0.5` after antialiased downsampling. The thin spear/appendage remains represented in RGBA, but much of its antialiased alpha falls below `0.5`.

For V0:

- native1024 alpha bbox span at threshold 0.5: `916 x 710`;
- R512 visible/RGB extent is approximately half-scale (`~463 x 360`);
- R512 alpha bbox at threshold 0.5 contracts to only `324 x 354`.

Therefore this is a **resolution-dependent support-threshold formulation failure**, not render corruption and not a P geometry failure.

## Current next gate — native-scale-once closure

Do not tune a lower alpha threshold post-result and do not relax P <= `0.005`.

The next optimizer-zero formulation must be:

```text
native ordered RGBA
    -> estimate h_sheet ONCE before resize
    -> carry h_sheet through 512/256 preprocessing
    -> model/features predict only depth
    -> analytic full P reconstruction
```

Do **not** re-estimate `h_sheet` from degraded 512/256 alpha.

Freeze a new optimizer-zero prereg on the same 16 FIT-only sentinel panel and same P p95 <= `0.005` gate. Reuse the exact native-derived scalar at 1024/512/256 reconstruction.

No TUNE/CAL/DEV/EXTERNAL split may be opened.

Only if that closure passes may a separate FIT-only depth/P overfit prereg authorize optimizer steps.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
