# IRIS Single-Pose V2 — Mini Extractability / Generalization Prereg V1

Date: 2026-08-24  
Status: `FROZEN_BEFORE_OPTIMIZER__OPEN_SPLITS_ONLY`

## Question

Given eight ordered single-pose RGBA observations, can the current IRIS V2 architecture learn the observable evidence that the frozen Representation Authority gate established as sufficient, and generalize that extraction to unseen OPEN TUNE assets?

Primary evidence under test: P common/object-frame surface position; Z_coarse high-recall persistent-locus basin evidence; Z_fine local correspondence refinement inside an admitted basin; U_geo geometry-risk evidence, diagnostic/calibration only in this mini.

N remains a learned dense local orientation field because `geom_n` is defined at the same raster observation locus as `geom_p`. CI104 did not establish `track_n_view` as exact persistent-locus correspondence authority. Therefore N is trained from `geom_n` and reported in angular degrees, but is forbidden from coarse admission, correspondence ranking, reciprocal/cycle truth and checkpoint selection. No R1/R3 normal score is reused as a learner promotion target.

## Prior authority

CI104 closed the immediate representation question with `P_GEOMETRY_SUFFICIENT`: 256 OPEN assets; 12,288 queries per exact arm; exact P top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical error median/p90/p95/max=0. The R2 P-noise curve is the descriptive robustness reference. No post-result change to R0-R3 is part of this mini.

## Architecture

Exact current `IRISSinglePoseV2` default config is frozen: shared f2/f4/f8/f16 image encoder; full-resolution f16 within-view axial reasoning; fixed 16x16 cross-view context with known yaw and resolution-independent x positional encoding; context fused back into local f16; P/N/U_geo/Z_fine at R/2; Z_coarse at R/8.

At mini input R=256: P/N/U_geo/Z_fine=128x128, Z_coarse=32x32. This is an extractability/generalization feasibility gate, not a claim of final native-1024 localization quality.

## Matcher / consumer roles

Global basin admission is `Z_coarse + predicted P` only. `Z_fine` is local-only refinement, never global admission or cross-basin reorder. N is forbidden from correspondence admission/ranking. Output remains top-k/set-valued. No mandatory learned ambiguity/singleton head is introduced. Reciprocal/cycle remain deterministic support evidence.

## Membership

Source representation-seed SHA-256: `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af`.

Source ordered panel-ID digest: `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`.

Mini membership SHA-256: `ef19120e2cc98fac50a0c94ba863fd3cc4cec4a8392b9c48684f96ffeba3ba50`.

Frozen roles: FIT_TRAIN=128; FIT_SELECT=32; TUNE_FINAL=26 (all TUNE assets in the frozen CI104 panel); 70 remaining FIT assets unused. Roles are asset-disjoint. CAL, DEV and EXTERNAL_HOLDOUT remain unopened.

Rare FIT source/capability strata represented by <=3 assets are forced into FIT_TRAIN before deterministic SHA-based selection of dominant strata. FIT_SELECT is held out from gradient updates. TUNE_FINAL is never used for checkpoint selection.

## Image / style policy

Input resolution=256. Native 1024 RGBA is validated; the existing canonical 512 derivative is used before PIL RGBA bilinear 512->256. Both `cel_clean` and `ink_cel` are staged. Physical firewall retains only vertices/faces as geometric source truth.

Training style is a deterministic per-asset/per-epoch choice between cel_clean and ink_cel. Evaluation runs both styles independently and equal-macro. No additional augmentation is authorized in V1.

## Truth / cache

Per asset: geometry samples=1024/view; anchor samples=512/view; max persistent tracks=2048; visibility radius=3 native pixels; max surface witness error=0.003 canonical units.

`track_xy` is exact continuous camera projection after a legal raster-surface visibility witness. `geom_n` is observation-local exact raster surface orientation and may train N. `track_n_view` is a nearby visibility-witness diagnostic and may not define correspondence positives or ranking.

Learner cache must pass the CI96 partial-visibility invariant: `track_visible.shape == track_surface_error.shape`; visible entries finite; hidden entries +inf; cache fingerprint binds `prepare_cache.py`, `geometry.py`, and `coords.py`.

## Training

Random seed=20260824. AdamW lr=3e-4, weight decay=1e-4, betas=(0.9,0.95), 16 epochs, microbatch=1 asset/eight views, gradient accumulation=4, clip=1.0, FP16. Train track samples=128; evaluation track samples=512. Epochs1-3 train P/N/U_geo only. From epoch4 add coarse multi-positive, bidirectional pair objective, hard margin, soft reciprocal, P cross-view consistency and local Z_fine. Candidate checkpoints are epochs4/8/12/16.

`Zc_reciprocal_soft` remains 0.02. No new ambiguity head is introduced. Fine track sampling is deterministic position-uniform thinning; prefix truncation is forbidden.

## GPU capacity release condition

Before scientific optimizer step1 the exact bundle must run a CUDA preflight at R=256 with current production-width model, microbatch1x8, full post-warmup loss graph, forward+backward, finite gradients, and AdamW two-moment memory accounted on GPU, while taking zero scientific optimizer steps. OOM/non-finite => `APPARATUS_CAPACITY_REOPEN_REQUIRED`; learner optimizer must not start.

The already-closed 256/512/1024 architecture executable preflight is not repeated as a scientific gate.

## FIT_SELECT evaluation

Checkpoint selection uses FIT_SELECT only, both styles. Report P Euclidean mean/median/p90/p95; N angular mean/median/p90/p95 diagnostic; U_geo risk diagnostics; Z_coarse exact containing-cell Recall@1/4/8/16/32; predicted-P coarse basin Recall; oracle-truth-basin Z_fine native-1024 localization; end-to-end retained-basin localization/PCK; reciprocal support; family tails; style non-collapse.

FIT_SELECT query budget=18 per asset/style, with adjacent/skip-one/opposite strata through the frozen evaluator. Qualification budget=6 per asset/style.

## Frozen checkpoint selection key

Lower is better:

`P_p95/0.005 + max(0,1-Zc_top8)/0.10 + oracle_Zf_top1_p95_native_px/16 + max(0,0.75-family_Zc_top8_p10)/0.25 + family_P_p95_p90/0.0065 + max(0,0.85-min_style_Zc_top8)/0.15 + max_style_P_p95/0.0065`.

Ties choose the earlier candidate epoch. N is explicitly absent. Random initialization is evaluated on FIT_SELECT as diagnostic baseline only. TUNE_FINAL remains untouched until checkpoint freeze.

## Final TUNE evaluation

After checkpoint selection evaluate exactly one frozen checkpoint on all 26 TUNE_FINAL assets, both styles. Query budget=24 per asset/style; qualification budget=8. No reselection or threshold change after TUNE is opened.

## Frozen outcome labels

Core FIT_SELECT extractability: P p95<=0.005; Z_coarse Recall@8>=0.90; oracle-basin Z_fine top1 p95<=16 native-1024 px.

TUNE core additionally: family Z_coarse Recall@8 p10>=0.75; minimum style Z_coarse Recall@8>=0.85; maximum style P p95<=0.0065.

Generalization gap: TUNE P p95<=max(0.0065,1.5*FIT_SELECT P p95) and TUNE Z-coarse Recall@8>=FIT_SELECT Z-coarse Recall@8-0.10.

Allowed labels: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

Thresholds/labels are frozen before optimizer use and are engineering promotion criteria, not information-theoretic claims.

## Non-claims

A PASS does not establish final 1024 local precision, autonomous source-track-free dense evidence construction, complete SurfaceBuilder, Geppetto/Arachne sufficiency, deformation-proof compilation or artist-domain robustness. A failure does not reopen R0 representation existence unless independent evidence invalidates the learner target; it first localizes to learner extractability, generalization or the named consumer submodule.
