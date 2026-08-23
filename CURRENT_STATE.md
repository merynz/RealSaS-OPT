# RealSaS-OPT — Current State

**Date:** 2026-08-23  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `REPRESENTATION_PASS__OPEN64_LEARNER_PASS__DENSE_M256_V1_2_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority. If a new chat says only `devam et`, continue from the exact next action below.

## Scientific target

**Observable evidence was not pruned; the problem was pruned.**

IRIS legal evidence remains: P, geometric N, support/visibility, U/risk, coarse/fine persistence, reciprocal/cycle consistency, set-valued ambiguity and provenance.

Forbidden IRIS target authority: hidden joint/owner IDs, parent graph, skin weights, pose-B mechanics, GFDR hidden mechanics.

## Corpus

Canonical selected: 3993. Preserved exclusions: 56 repair-pending + 6 active non-Basis shape-key + 1 blank. Controlled V1 usable: **3930**.

Frozen split:

- FIT 2935
- TUNE 313
- CAL 246 — SEALED
- DEV 270 — SEALED
- EXTERNAL_HOLDOUT 166 — SEALED

No resplit.

## Representation ceiling — PASS

64 deterministic open assets / 65,830 pairs:

- exact P top1 `0.9976302598`
- exact P top4 `1.0`
- exact P top8 `1.0`
- P noise .0025 top4/top8 `1.0 / 1.0`
- P .005 + N 20° top4 `0.9988151299`, top8 `0.9996658059`
- ambiguity within 0.003 = `0.0151754519`

Interpretation: observable addressability exists strongly. Exact/modest-noise P already preserves truth completely in top4; top8 is a hard-tail safety envelope.

## Open64 learner pilot — PASS

51 FIT / 13 TUNE, 8 epochs:

- TUNE P error `0.6096198788 -> 0.1311205992` = 78.49% reduction
- TUNE N error `0.9678560908 -> 0.1776760645` = 81.64% reduction
- raw Z_coarse top8 `0.1243990385 -> 0.9651442308`
- raw Z_fine top8 `0.1039663462 -> 0.9375`
- raw coarse top4 `0.8957331731`
- raw fine top4 `0.8527644231`

These were raw descriptor retrieval metrics, not final matcher metrics.

## Matcher lineage clarification

A sparse `iris_evidence_matcher_v1.py` also exists in repo. It has useful FIT-only calibration, reciprocal and cycle logic, but its candidate nodes are built from cached `track_visible/track_xy` physical-track anchors. GT track IDs are not used in its score, yet the candidate-node universe itself is truth-derived. Therefore it is retained as **diagnostic/ablation lineage**, not the canonical full-inference M256 evaluator.

D3 is still **not implemented** and remains reserve-only.

## Canonical D3-free dense matcher

Canonical current matcher:

`experiments/iris_controlled_v1/iris_dense_matcher_v1.py`

Dense evaluator:

`experiments/iris_controlled_v1/evaluate_iris_dense_matcher_v1.py`

Candidate domain is **alpha-supported dense 128×128 image-grid pixels**. GT physical tracks are used only to select evaluation queries and score target coordinates; they are never the candidate list or matcher authority.

Pipeline:

```text
alpha-supported dense target pixels
∩ known-camera row corridor
→ Z_coarse top16
∪ predicted-P nearest top4
→ scale-free rank fusion of Z_coarse + Z_fine + predicted-P
→ final ordered top8
→ reciprocal top1/top4 in image coordinates
→ two-third-view cycle support in image coordinates
→ U-risk qualification
→ confident singleton / top4 ambiguity / top8 ambiguity
```

N is not forced into the correspondence ranking with an uncalibrated coefficient; it remains rich observable/downstream geometry evidence. U cannot create a singleton; it may only widen the output set.

## Fresh M256 dense gate

Canonical prereg:

`experiments/iris_controlled_v1/IRIS_CONTROLLED_V1_M256_PREREG_V1_1.md`

Membership:

- 208 FIT
- 48 TUNE
- total 256
- all deterministic ceiling/open64 pilot assets excluded (`pilot64_overlap = 0`)
- 12 epochs from scratch
- corrected trainer `train_iris_controlled_v1_v1_1.py`
- CAL/DEV/EXTERNAL closed.

Decision localization:

- `LEARNER_FAIL`
- `MATCHER_CONSUMER_FAIL_PARTIAL`
- `M256_PASS`

Frozen matcher/safety gate includes dense fused top8 >= .97, adaptive output-set truth coverage >= .97, p10 fused top8 >= .90, and confident singleton precision >= .95 at >= .10 coverage.

## Canonical M256 runtime — USE V1.2 ONLY

Authority manifest:

`experiments/iris_controlled_v1/M256_DENSE_AUTHORITY_POINTERS_V1_2.json`

Runner:

`run_iris_controlled_v1_m256_v1_2.py`

SHA-256:

`72aa97e1902e831e5b87f054fceb187c04de55c7948a5ff19465d55b700249ab`

Notebook:

`IRIS_CONTROLLED_V1_M256_V1_2.ipynb`

SHA-256:

`21fcf07de2e25a194c4e84fd7f9a23124a9946c920616d912928c89ba8560cea`

Dense matcher SHA-256:

`6d70c42439e82fc02a4893e49965d206d02a05bd6d505fb42904f2a5b9fd4248`

Dense evaluator SHA-256:

`ba24f0a5b010f29c793c0a9eabbde02892a7c7841937c54babd634ee3d8e1f6b`

All canonical v1.2 runtime files were Google Drive round-trip SHA verified; Python entrypoints compile; dense matcher synthetic self-test PASS.

Earlier M256 runtime entries (`run_iris_controlled_v1_m256.py`, v1.1 runners/notebooks) are **SUPERSEDED BEFORE EXECUTION** and retained only as provenance.

## NEXT EXECUTABLE STEP

Use NVIDIA Colab. From the package directory run:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python run_iris_controlled_v1_m256_v1_2.py --workers 4 --epochs 12
```

It will:

1. SHA-verify base package + fast prep + corrected trainer + dense matcher/evaluator;
2. deterministically select fresh 208 FIT + 48 TUNE excluding pilot64;
3. prepare only those 256 open assets to local SSD;
4. run dense matcher at identical random-init witness;
5. train 12 epochs from scratch;
6. run dense matcher on best TUNE-selected checkpoint;
7. write final decision to:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_CONTROLLED_V1_M256_V1_1/M256_DECISION.json`

## Authorization state

`DENSE_M256 = AUTHORIZED_AND_NEXT`

`1024_INTERMEDIATE = NOT_AUTHORIZED_UNTIL_DENSE_M256_PASS`

`FULL_3248_PRODUCTION = NOT_AUTHORIZED`

`D3 = NOT_IMPLEMENTED__RESERVE_ONLY`

`CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

## VERY IMPORTANT M4 audit

Authority:

`experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

The downstream E0–E5 question remains mandatory: can richer observable/set-valued evidence form a RigAnything-like 2.5D equivalent substrate sufficient for Geppetto and functional rigging?

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> only then information limit`
