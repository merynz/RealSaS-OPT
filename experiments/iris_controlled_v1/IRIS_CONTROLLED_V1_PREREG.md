# IRIS Controlled V1 — Preregistration

**Frozen before optimizer steps.**  
Date: 2026-08-23

## Question

Can an image-only multi-view learner recover persistent observable surface geometry from eight controlled neutral-pose views when exact cameras and exact legal surface truth are available?

A failure is **not** an information-impossibility result unless apparatus, representation and learner failure classes have been separated.

## Data

Usable controlled corpus: 3,930 assets, frozen family-disjoint split membership:

- FIT 2935 — optimizer access;
- TUNE 313 — model selection;
- CAL 246 — sealed;
- DEV 270 — sealed;
- EXTERNAL_HOLDOUT 166 — sealed.

Excluded but preserved: 56 repair-pending, 6 active-shape-key, 1 all-8 blank.

Training inputs use `cel_clean_512` / `ink_cel_512` observations. Style is sampled during FIT; TUNE uses the fixed clean style for stable selection. The default neural input resolution is 256 while exact supervision is expressed in normalized image coordinates and canonical physical coordinates.

## Legal truth

Allowed: RGB/alpha, ordered view/yaw metadata, triangle+bary raster authority, canonical P, geometric N, visibility/support, cross-view physical-locus relation, uncertainty targets derived from prediction error, provenance.

Forbidden as IRIS input/target authority: joint IDs, bone roles, parent graph, owner/skin identity, dense skin weights, pose-B mechanics, GFDR fields, teacher mechanical slots.

## Model

`IRISControlledV1`: shared multiscale encoder -> within-view axial reasoning -> row-constrained multi-view transformer -> decoder -> P/N/U/Z_coarse/Z_fine.

Deterministic wrapper owns top-k, reciprocal/cycle, ambiguity-set policy, support, fusion, reprojection and provenance.

## Optimization

- epochs: 24
- batch: 1
- AdamW lr: 5e-5
- weight decay: 1e-4
- grad clip: 2.0
- seed: 20260823
- P loss: SmoothL1
- N loss: 1-cosine
- U: heteroscedastic Laplace-like risk calibration with detached P error
- epochs 0..3: P/N/U only
- epoch >=4: add coarse multi-positive NCE, fine multi-positive NCE, P cross-view consistency

Loss weights after warmup:

`L = P + .25 N + .05 U + .10 Zc + .05 Zf + .20 P_consistency`.

## Mandatory no-optimizer gates

1. package SHA verification;
2. deterministic seed generation agrees with frozen split membership;
3. cache construction succeeds on open splits;
4. exact representation ceiling on 64 deterministic open assets:
   - P exact top8 >= .995;
   - P+N exact top8 >= .995;
   - per-family P-exact top1 p10 >= .95;
5. real-asset executable preflight already PASSes cache -> model -> loss -> backward.

If gate 4 fails, do not train; diagnose representation/apparatus first.

## Selection and sealed policy

Checkpoint selection uses TUNE total loss plus diagnostic P/N/Z metrics. No CAL/DEV/EXTERNAL access is authorized by the training launcher. `evaluate_iris_controlled_v1.py` fails closed on sealed panels unless `--authorize-sealed` is explicitly supplied.

A TUNE gain cannot be described as product generalization. Sealed opening requires a separate decision after the controlled training run and must be recorded before results are viewed.

## Failure interpretation

- poor P/N with exact ceiling PASS -> learner/optimization/architecture failure;
- good P/N but poor matching -> persistence/evidence-layer failure;
- exact representation ceiling failure -> representation/target/apparatus investigation;
- only after stronger legal representations fail may genuine observation non-identifiability be claimed.
