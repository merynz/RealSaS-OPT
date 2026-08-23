# IRIS Controlled V1 — Training-Ready Report

**Date:** 2026-08-23  
**Status:** `TRAINING_READY__OPTIMIZER_NOT_STARTED__SEALED_PANELS_CLOSED`

## Corpus authority

The immediate controlled IRIS corpus contains **3,930** assets from the 3,993 selected master families. The 63 excluded assets are preserved, not deleted:

- 56 Stage-B6 repair-pending assets whose frozen corrected geometry does not yet have matching canonical A-pass rerenders;
- 6 active non-Basis shape-key assets with unresolved canonical rest-state authority;
- 1 all-eight-views blank observation.

Split membership is inherited unchanged from the master corpus:

- FIT 2935
- TUNE 313
- CAL 246
- DEV 270
- EXTERNAL_HOLDOUT 166

Open optimizer/selection population: 3248. Sealed population: 682.

## Source authority hashes

The freeze is derived from and verifies:

- `CANONICAL_VARIANT_SELECTION.json` — `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`
- `IRIS_TRAINING_ELIGIBILITY_V1.json` — `b4ee5b87b668ee2a2071e0a26a3d1e6bd0adcf3563bffe88237a0ce7f8dab717`
- `POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_RESULT_V1.json` — `a86178b46dd55636356d27c2f1fe6d042aea1ea7863ecfc2bf33e7a9a0f868f9`

The 56 deferred repair artifacts retain individual normalized-geometry and source-structure SHA-256 values in the Drive authority `IRIS_CONTROLLED_V1_DEFERRED_REPAIR_SHA_REGISTRY.json`.

## Consumer/export correction

The legacy `exports/IRIS/records.jsonl` is not used as training authority because its current production export contains only 11 rows. Controlled V1 builds its deterministic seed from the canonical selection + frozen 3930 split membership, and then reads directly from:

`master/assets/<asset>/primary_geometry.npz`

and

`master/assets/<asset>/renders/V0..V7/{camera.json,raster_authority.npz,cel_clean_512.png,ink_cel_512.png}`.

This removes an incomplete export layer from the critical path without changing master truth.

## Exact observation truth

Cross-view truth is constructed only from legal observable geometry:

`pixel -> triangle_id + barycentric_uv -> canonical continuous surface P`.

Geometric N is reconstructed from canonical vertices/faces. Multi-view tracks are accepted by camera projection plus local raster search plus physical surface-error qualification. Mechanical owner/bone/parent/skin truth is absent from this path.

## Executable preflight

Real FIT asset: `asset_00027381105293114a49dc90`.

Observed results:

- 8/8 cameras/raster authorities loaded;
- 128 cross-view physical tracks built;
- track support histogram: `{4: 122, 5: 6}`;
- neural core trainable parameters: 5,657,863;
- P/N/U/Z forward shapes valid;
- full loss finite;
- backward gradient absolute sum > 0;
- preflight PASS.

The associated JSON is `IRIS_CONTROLLED_V1_REAL_ASSET_PREFLIGHT.json`.

A one-real-asset representation-ceiling apparatus smoke also PASSed: exact P and exact P+N achieved top1/top8 = 1.0 across 390 cross-view pairs. This is not substituted for the mandatory 64-open-asset pre-optimizer ceiling; the launcher runs that automatically after cache preparation.

## Frozen model/system boundary

Learned neural core:

`SharedImageEncoder -> AxialWithinViewReasoning -> RowWiseMultiViewFusion -> dense decoder -> P/N/U/Z_coarse/Z_fine`.

Deterministic IRIS evidence layer owns candidate restriction, top-k, reciprocal/cycle checks, local reranking, ambiguity retention, observation support, geometry fusion/reprojection and provenance.

**The observable evidence was not pruned. The problem was pruned.** Controlled V1 retains a richer observation contract while removing the requirement that IRIS infer authored hidden mechanical identity.

## Default training contract

- 8 ordered views
- frozen 512 observation source; default model input 256
- 24 epochs
- batch 1
- AdamW 5e-5
- weight decay 1e-4
- grad clip 2.0
- geometry-only warmup through epoch 3
- correspondence objectives from epoch 4
- FIT train / TUNE selection only
- CAL, DEV, EXTERNAL_HOLDOUT fail closed unless separately authorized

## Remaining work that requires execution rather than design

No further architecture/corpus design work is required before the controlled run. The compute sequence is now mechanical:

1. `launch_iris_controlled_v1.py --mode prepare`
2. `launch_iris_controlled_v1.py --mode ceiling`
3. `launch_iris_controlled_v1.py --mode train`

This is deliberately not described as product readiness. Appearance-domain B/C renders, camera jitter/unknown-camera robustness, and the 56-asset repair publication remain later work after the controlled question is answered.
