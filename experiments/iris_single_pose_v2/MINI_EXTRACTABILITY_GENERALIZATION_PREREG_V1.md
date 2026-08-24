# IRIS Single-Pose V2 — Mini Extractability / Generalization Prereg V1

Date: 2026-08-24  
Status: `FROZEN_BEFORE_OPTIMIZER__OPEN_SPLITS_ONLY`

## Question

Given eight ordered single-pose RGBA observations, can the current IRIS V2 architecture learn the observable evidence that the frozen Representation Authority gate established as sufficient, and generalize that extraction to unseen OPEN TUNE assets?

Primary evidence: P common/object-frame surface position; Z_coarse persistent-locus basin evidence; Z_fine local refinement inside an admitted basin; U_geo geometry-risk diagnostic. N remains a dense local orientation target because `geom_n` is defined at the same raster observation locus as `geom_p`, but CI104 did not establish `track_n_view` as persistent-locus correspondence authority. N is therefore forbidden from coarse admission, correspondence ranking, reciprocal/cycle truth and checkpoint selection.

## Prior authority

CI104: `P_GEOMETRY_SUFFICIENT` on 256 OPEN assets / 12,288 exact-P queries: top1/top4/top8=1/1/1, reciprocal=1, cycle=1, physical-error median/p90/p95/max=0. R2 P-noise remains the descriptive robustness reference.

## Architecture

Current `IRISSinglePoseV2` default config is frozen: shared f2/f4/f8/f16 encoder; full-resolution f16 axial within-view reasoning; fixed 16x16 cross-view context with yaw and resolution-independent x position; context fused into local f16; P/N/U_geo/Z_fine at R/2; Z_coarse at R/8. At R=256 these are 128x128 and 32x32 respectively. This mini is not a final 1024-precision claim.

## Matcher roles

Global admission=`Z_coarse + predicted P`; Z_fine=local-only refinement; N forbidden from correspondence ranking; top-k/set-valued output retained; no mandatory learned ambiguity head; reciprocal/cycle remain deterministic support evidence.

## Membership

CI104 representation-seed byte SHA-256: `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af`. It is prior provenance only and is **not a runtime input** to this learner mini.

Source ordered panel-ID digest: `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`.

Mini membership **canonical JSON** SHA-256: `4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055`. Canonical JSON means parsed JSON serialized with sorted keys and compact separators; packaging byte SHA is independently bound by bundle `SHA256SUMS.txt`.

`MINI_MEMBERSHIP_PROVENANCE_V1.json` is a frozen pre-run proof generated against the exact CI104 representation seed. It binds the source-seed SHA, panel digest, membership SHA and role-ID digests and certifies that all 128 FIT_TRAIN + 32 FIT_SELECT IDs were source split FIT and all 26 TUNE_FINAL IDs were source split TUNE. Source registry/capability metadata is not carried into the runtime seed.

Frozen roles: FIT_TRAIN=128; FIT_SELECT=32; TUNE_FINAL=26 (all TUNE assets from the frozen CI104 panel); unused FIT=70. Roles are asset-disjoint. CAL/DEV/EXTERNAL_HOLDOUT remain unopened. Rare FIT source/capability strata count<=3 were forced into FIT_TRAIN before deterministic SHA-based dominant-stratum selection during membership freeze; this stratification metadata is not consumed by the learner runtime. FIT_SELECT receives no gradients. TUNE_FINAL is never used for checkpoint selection.

**Hard TUNE/runtime firewall:** the learner runtime seed is built only from frozen membership and contains exactly `asset_id`, `split`, and `mini_role`; the CI104 representation seed is not opened by the runtime. Before checkpoint selection is frozen, no TUNE_FINAL RGBA, raster authority, camera, derived truth cache, or TUNE cache manifest may be staged, decoded, hashed for learner apparatus, or evaluated. Pre-optimizer TUNE knowledge is therefore restricted to the already-frozen asset-ID membership and its FIT/TUNE role label. After `CHECKPOINT_SELECTION_FROZEN.json` exists, the exact 26 TUNE_FINAL assets may be staged/cached once for the single final evaluation.

## Image/style policy

Input=256. Native1024 RGBA is validated; the existing canonical512 derivative is used before PIL RGBA bilinear 512->256. Both cel_clean and ink_cel are staged. Training style is deterministic per-asset/per-epoch random choice; evaluation runs both separately and equal-macro. No other V1 augmentation.

## Truth/cache

Per asset: geometry samples1024/view; anchors512/view; max tracks2048; visibility radius3 native px; max witness error0.003. `track_xy` is exact continuous projection after a legal raster witness. `geom_n` is observation-local raster orientation and may train N. `track_n_view` is witness diagnostic only. Cache requires visible/error shape equality, finite visible errors, +inf hidden errors, and fingerprints prepare_cache.py+geometry.py+coords.py.

## Training

Seed20260824; AdamW lr3e-4, wd1e-4, betas(.9,.95); 16 epochs; microbatch1; grad accumulation4; clip1; FP16; train tracks128; eval tracks512. Epochs1-3 P/N/U_geo warmup; from epoch4 add coarse multi-positive, bidirectional pair objective, hard margin, soft reciprocal, P consistency and local Z_fine. Candidate checkpoints4/8/12/16. `Zc_reciprocal_soft=0.02`. Fine track sampling is deterministic position-uniform, never prefix truncation.

## GPU release condition

Before scientific optimizer step1, exact bundle must pass CUDA R=256 production-width full post-warmup loss forward+backward, finite gradients, and AdamW two-moment memory accounting with zero scientific optimizer steps. OOM/non-finite => `APPARATUS_CAPACITY_REOPEN_REQUIRED`. The already-closed 256/512/1024 architecture executable preflight is not repeated as a scientific gate.

## FIT_SELECT evaluation / checkpoint key

Both styles. Query budget18/asset/style; qualification6. Report P tails, N angular diagnostic, U risk diagnostics, Z_coarse Recall@1/4/8/16/32, predicted-P basin retrieval, oracle-basin Z_fine native-1024 localization, end-to-end localization/PCK, reciprocal support, family/style tails.

Frozen lower-is-better selection key:

`P_p95/0.005 + max(0,1-Zc_top8)/0.10 + oracle_Zf_top1_p95_native_px/16 + max(0,0.75-family_Zc_top8_p10)/0.25 + family_P_p95_p90/0.0065 + max(0,0.85-min_style_Zc_top8)/0.15 + max_style_P_p95/0.0065`.

Tie -> earlier epoch. N is absent. Random-init FIT_SELECT is diagnostic. Phase A accepts only FIT_TRAIN/FIT_SELECT cache paths and has no TUNE CLI argument. TUNE images/truth remain untouched until `CHECKPOINT_SELECTION_FROZEN.json` is written.

## Final TUNE / frozen labels

Only after the FIT checkpoint freeze, stage/cache exactly the 26 frozen TUNE_FINAL assets, audit that TUNE-only apparatus, then evaluate exactly one selected checkpoint on all26 TUNE_FINAL assets, both styles; 24 queries/asset/style, qualification8. No reselection/threshold changes. The run authority must record `tune_stage_started_after_checkpoint_freeze=true` and `representation_seed_consumed_at_runtime=false`.

FIT_SELECT core: P p95<=0.005; Zc Recall@8>=0.90; oracle Zf top1 p95<=16 native px.

TUNE additionally: family Zc@8 p10>=0.75; min-style Zc@8>=0.85; max-style P p95<=0.0065. Generalization gap: TUNE P p95<=max(0.0065,1.5*FIT_SELECT P p95) and TUNE Zc@8>=FIT_SELECT Zc@8-0.10.

Allowed labels: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

These are frozen engineering promotion criteria, not information-theoretic claims.

## Non-claims

PASS does not establish final1024 local precision, autonomous source-track-free dense evidence, SurfaceBuilder, Geppetto/Arachne, deformation-proof compilation or artist-domain robustness. A learner failure does not reopen exact-P information existence unless independent evidence invalidates the target.
