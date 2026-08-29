# RealSaS — Controlled DINOv2 Representation Ladder Preregistration V1

**Date:** 2026-08-29  
**Status:** `SEALED_BEFORE_DINO_CANDIDATE_OUTPUTS__APPARATUS_PREFLIGHT_REQUIRED__TRAINING_NOT_YET_AUTHORIZED`  
**Parent canonical main:** `b25dc36972e156c669e721fdddb5f1466b9eabf4`

## Plain-language question

With the training population, training exposure, raster framing, native-detail path, fusion/head capacity, optimizer protocol, target and downstream evaluation held fixed, does a stronger frozen DINOv2 representation make the required RealSaS camera-forward depth field easier to learn accurately?

This is a **frozen representation accessibility experiment**, not an asymptotic-capacity proof and not a product-readiness claim.

## Why this gate exists

The historical 32 -> 128 -> 512 family experiment cannot answer this question because total optimizer steps were fixed at 7168 while family count changed. Nominal exposure therefore fell from 1792 to 448 to 112 family selections per family. Unseen transfer improved while training precision degraded. The new ladder removes that confound by using one exact training population and one exact exposure stream for every representation rung.

The already-closed sacrificial consumer interlock establishes that the RealSaS 2.5D substrate has a genuine downstream skeleton/skin/Compiler consumer path. It does not turn G0/A0 into product models and does not redefine the depth-tolerance object.

## Scientific candidates — the only intended treatment variable

Frozen extractors:

- `S`: official DINOv2 ViT-S/14 family, token width 384;
- `B`: official DINOv2 ViT-B/14 family, token width 768;
- `L`: official DINOv2 ViT-L/14 family, token width 1024;
- `g`: official DINOv2 ViT-g/14 family, token width 1536.

All foundation parameters are frozen. No backbone fine-tuning is allowed in this phase.

Before optimizer step 0, apparatus preflight MUST seal the exact upstream source revision/model identifier and SHA-256 of every resolved weight file. If any weight authority cannot be reproduced exactly, training remains closed.

## Observation contract and framing

Scientific input mode is ObservationContract V1 Mode G only:

- exactly 8 ordered views;
- native `1024 x 1024 RGBA`;
- exact orthographic cameras;
- native alpha/support remains observation authority;
- no unmodelled camera error;
- no adaptive crop, per-asset zoom or per-view recrop.

Foundation input is the stored RGB observation on the full authoritative canvas, deterministically antialiased-resized once to `518 x 518`. Since `518 = 37 x 14`, every rung sees the same `37 x 37` patch-14 grid.

The shared native-detail/support path remains `1024 x 1024` and must be byte/config identical across all rungs. Alpha is not promoted into a second learned geometry authority.

## Exact training population — one population for all four rungs

Reuse the exact historical controlled **512-FIT-family training membership** from the P-V5 family-scale ladder, but consume the corresponding native-1024 production observations.

Historical membership authority:

- source metadata authority SHA-256: `475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`;
- deterministic training seed: `PV5_R256_FIT_SCALE_TRAIN_V1`;
- source quota: `483 Objaverse / 24 Quaternius / 5 KayKit`;
- styles per family: `cel_clean` and `ink_cel`;
- complete membership canonical JSON SHA-256: `8531360f1c61dc4cdb699c095790c35ab3af0da5a7a290b420e5777bb4e9169b`.

Apparatus preflight must reconstruct/recover that exact membership and verify the canonical membership hash. It must then establish 512/512 availability in `RealSaS_MASTER_CORPUS_1024_V3`.

**No missing family may be replaced.** If any member cannot be mapped to the authoritative native-1024 corpus, the preflight is FAIL and this preregistration must be versioned before training; no outcome-sensitive substitution is permitted.

All four DINO rungs consume the exact same 512 families, both styles, same view ordering and same target loci.

## Held-out evaluation population

Reuse the historical `FIT_PROXY32` membership encoded by the same frozen membership authority:

- 32 FIT families;
- both `cel_clean` and `ink_cel` styles;
- zero gradients;
- disjoint from the complete 512-family training set;
- never used for optimizer scheduling, stopping, learning-rate choice or architecture modification.

`DEV32`, CAL, EXTERNAL_HOLDOUT and later sealed sets remain closed to this experiment.

The historical nested `TRAIN_DIAG32` set is retained only as a training-fit diagnostic and never selects a checkpoint.

## Geometry target — one learned authority

The only required learned geometric output is camera-forward depth `d`.

At every authoritative supervision/evaluation locus:

`d_truth = dot(P_truth - O, F)`

and

`P_hat = O + d_hat F`.

Training/evaluation loci are the frozen deterministic 4096 visible raster-authority samples per view, derived from authoritative canonical geometry/barycentric provenance. The same loci are used for every rung and for both styles of a family.

No skeleton, skin, mechanical owner, source-rig identity or product-canonical ID may enter the learner.

No learned normal, uncertainty/risk head or camera head participates in this Phase-1A ladder. Deterministic `N_d` may be derived downstream exactly as in the frozen DTB-ND1 route when a consumer requires it.

## Removing the prospective risk-band confound

The prospective V4 design proposed candidate-native risk replay plus a common band. The later prospective seal already records that a spatially constant `kappa_common * D_hull` band can be degenerate.

For this representation ladder V1, **risk calibration is removed from the scientific treatment entirely**:

- candidates produce depth only;
- there is no candidate-specific risk head;
- there is no constant common-band ranking path;
- representation ranking cannot be rescued or harmed by risk-calibration behavior.

Risk/abstention calibration is a separate later experiment after a representation candidate is selected.

## Fixed width interface — no trainable adapter advantage

For a DINO token `x in R^d`, use the fixed isometric injection:

`Q_d(x) = [x, 0_(1536-d)]`.

Thus S/B/L are zero-padded to 1536 channels and g is identity. `Q_d` contains no learned parameters and preserves Euclidean distances/norms before normalization.

After `Q_d`, apply one identical non-affine token normalization implementation to all candidates. The exact epsilon and implementation bytes are sealed by apparatus preflight.

No rung-specific trainable projection is permitted before the shared trainable stack.

## Shared trainable system

Everything after the frozen representation must be identical across S/B/L/g:

- native-1024 detail/support encoder;
- exact-orthographic camera adapter;
- 8-view structural fusion;
- dense/sampled depth decoder;
- loss construction;
- trainable parameter count;
- parameter initialization seed;
- optimizer/scaler semantics.

The exact architecture/config/code SHA and trainable parameter count must be sealed by apparatus preflight before any scientific optimizer step. If parameter counts differ by rung, preflight FAILS.

No candidate-specific dropout, hidden width, layer count, decoder, normalization or augmentation is allowed.

## Sample/order stream

One deterministic family/style/view/locus stream is generated once and its manifest/hash is shared by all rungs.

Each scientific optimizer step selects exactly **8 families**, and both styles are used for every selected family: `16 family-style cells / step` before any fixed microbatch/accumulation split.

Family selection uses deterministic balanced cycling/permutation over the same 512 members. No hard-example mining, loss-dependent resampling or rung-specific sampling is allowed.

If memory requires different physical microbatch sizes for different extractors, that is permitted only for **frozen feature extraction**. The trainable stack consumes cached tensors under an identical logical batch and update stream. Cached-vs-online token parity must pass at scientific optimizer step 0.

## Fixed optimizer/exposure budget

Every rung begins from the same trainable-module seed and follows the same logical schedule.

### Phase MAIN

- AdamW;
- learning rate `3e-4`;
- betas `(0.9, 0.95)`;
- weight decay `0`;
- logical updates `1..2048`.

### Phase TAIL

At logical update 2049, create fresh AdamW moments once for every rung:

- learning rate `3e-5`;
- betas `(0.9, 0.95)`;
- weight decay `0`;
- continue through logical update `32768` without another optimizer reset.

Two checkpoints are mandatory and neither is selected from outcomes:

1. `MATCHED_7168`: update 7168, reproducing the historical total-update budget for diagnostic continuity;
2. `PRIMARY_32768`: update 32768, the primary Phase-1A comparison checkpoint.

At 512 training families and 8 family selections/update:

- `MATCHED_7168` = 112 nominal family exposures;
- `PRIMARY_32768` = 512 nominal family exposures.

The primary budget was chosen before DINO candidate outputs because the old 112-exposure rung was globally underfit and the old 128-family rung had 448 exposures/family. This does **not** claim 512 exposures is asymptotic convergence; it defines a substantially less-starved, equal-budget comparison.

No candidate-specific early stopping, extension or learning-rate change is permitted after candidate outcomes are opened.

## Loss

Primary gradient-bearing loss is identical across rungs and supervises camera-forward depth at the frozen loci. Apparatus preflight must seal the exact FP32 loss implementation and any fixed scalar weights before optimizer step 0.

No DINO outcome may be used to add an auxiliary objective. Any future auxiliary normal/reprojection/silhouette loss is a new versioned experiment unless already identically sealed before candidate outputs.

## Primary scientific evaluation

Phase-1A asks whether a representation makes the required depth field accessible under the fixed RealSaS learner/budget.

For each rung and each mandatory checkpoint:

1. generate the actual predicted depth residual field on every held-out family/style across all 8 views;
2. sample/evaluate at the exact frozen authority loci;
3. run the actual residual field through the frozen DTB-ND1 persistence/consumer replay wherever its typed input contract applies;
4. report direct replay verdicts plus depth diagnostics;
5. never replace direct replay by a mean/median summary.

The conservative measured geometry target remains the already-closed DTB-ND1 lower safe endpoint:

- RMS reference: `0.00250`;
- matched ell=0 absolute-depth-P95 reference: approximately `0.00490`.

These numbers are **diagnostic target lines**, not a substitute for direct replay on arbitrary residual structure.

### Representation-ladder PASS label

A rung receives `REPRESENTATION_ACCESSIBLE_V1` only if, at `PRIMARY_32768`:

- no frozen replay/typed-route hard failure occurs on an attempted held-out cell;
- at least 95% of held-out family-style cells pass the frozen actual-residual replay criterion;
- the same rung is not globally underfit according to the preregistered training-fit diagnostic report.

The exact historical replay check set/threshold bytes must be sealed during apparatus preflight and cannot be modified from DINO results.

If the current replay implementation cannot legally produce a per-family verdict on this new residual carrier, apparatus preflight must stop before training and version the evaluator contract. It may not silently substitute the `0.00490` scalar threshold as product authority.

## Candidate ordering

Among rungs that receive `REPRESENTATION_ACCESSIBLE_V1`, order by:

1. higher held-out actual-residual replay pass coverage at `PRIMARY_32768`;
2. lower across-cell P95 camera-forward absolute-depth error;
3. lower across-cell P95 projected `||P_hat-P_truth||`;
4. lower deterministic delivered-`N_d` angular residual where defined;
5. smaller/faster frozen extractor.

A primary FAIL cannot be rescued by diagnostics, speed, model size or later OOD probes.

If no rung passes, report `NO_PASS_PHASE1A_V1`; do not reinterpret `g <= B` or all-four-fail as proof that representation capacity is irrelevant.

## Mandatory structure-scale diagnostics

Using the frozen native-1024 alpha distance-transform thickness operator, report for every rung:

- `SUBPATCH_THIN`: diameter `<28 px`;
- `ONE_TO_TWO_PATCH`: `28 <= diameter <56 px`;
- `BROAD`: diameter `>=56 px`.

Within each stratum report camera-forward depth residual, projected P residual, deterministic `N_d` angular residual where defined, and the preregistered high-frequency surface-energy retention diagnostic.

These strata explain *where* rung differences occur; they do not alter PASS.

## Required apparatus preflight before training opens

Training remains closed until one machine-readable preflight proves all of the following:

1. exact 512-family train membership reconstructed/recovered and canonical membership SHA matches;
2. exact FIT_PROXY32 / TRAIN_DIAG32 membership recovered and disjointness verified;
3. 512/512 train-family and 32/32 held-out native-1024 assets available with both frozen styles and all 8 views;
4. exact DINO weight authorities and SHA-256 hashes sealed;
5. 518 whole-canvas preprocessing and 37x37 token shapes identical;
6. fixed `Q_d` and normalization verified, including norm-preservation before normalization;
7. shared native-detail/fusion/depth-head bytes/config identical and trainable parameter counts exactly equal;
8. one deterministic sample/order manifest shared by all four candidates;
9. cached-vs-online frozen token parity passes on preregistered sentinel examples;
10. the frozen DTB-ND1 actual-residual replay accepts the new carrier without semantic reinterpretation;
11. no DEV32/CAL/EXTERNAL/sealed-family data opened;
12. scientific optimizer steps remain exactly zero.

Only after that preflight is sealed may a later execution decision authorize training.

## No-pass tree

- If training-fit diagnostics are poor for all rungs at 32768, localize shared learner/optimizer adequacy before representation conclusions.
- If training fit is adequate but all held-out rungs fail, Phase 1B frozen-plan fine-tuning becomes eligible as a separately preregistered intervention.
- If Phase 1B fails, full MapAnything/DA3 system comparison remains Phase 2.
- No cheap deterministic normal/hull/persistence retuning is opened from Phase-1A candidate outcomes.

## Firewalls

- DEV32 remains closed;
- FIT_PROXY32 gets no gradients and no checkpoint selection authority;
- no architecture/optimizer/membership edits after candidate outputs open;
- no candidate-specific risk head;
- no learned camera authority;
- no product-quality claim from sacrificial G0/A0;
- no source-rig/mechanics truth in IRIS;
- no declaration of capacity falsification from rung ordering alone.

## Execution state

This file seals the scientific design only. **DINO training is still not authorized.**

NEXT after promotion of this preregistration is one apparatus/data/weight-authority preflight with optimizer steps fixed at zero.