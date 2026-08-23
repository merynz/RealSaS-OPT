# RealSaS-OPT — Current State

**Date:** 2026-08-23  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `IRIS_REPRESENTATION_PASS__OPEN64_LEARNER_PASS__MATCHER_V1_READY__MINI256_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority. If a new chat says only `devam et`, continue from the exact next action below.

## Scientific target

**Observable evidence was not pruned; the problem was pruned.**

IRIS target:

`8 controlled neutral views -> persistent observable surface geometry + uncertainty`

Legal evidence: P, geometric N, support/visibility, uncertainty/risk, coarse/fine persistence, reciprocal/cycle consistency, set-valued ambiguity, provenance.

Forbidden target authority: hidden joint/owner IDs, parent graph, skin weights, pose-B mechanics, GFDR hidden mechanics.

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
- exact P family top1 p10 `0.9934295619`
- P noise .0025 top4/top8 `1.0 / 1.0`
- P .005 + N 20° top4 `0.9988151299`, top8 `0.9996658059`
- ambiguity within 0.003 = `0.0151754519`

Interpretation: observable addressability exists strongly. Top4 exact coverage is already complete in the controlled exact-P setting; top8 is a hard-tail safety envelope.

## Open64 learner pilot — PASS

51 FIT / 13 TUNE, 8 epochs:

- TUNE P error `0.6096198788 -> 0.1311205992` = 78.49% reduction
- TUNE N error `0.9678560908 -> 0.1776760645` = 81.64% reduction
- raw Z_coarse top8 `0.1243990385 -> 0.9651442308`
- raw Z_fine top8 `0.1039663462 -> 0.9375`

Important: these Z metrics are **raw descriptor retrieval**, not the final matcher. They do not include P geometry composition, reciprocal, cycle or ambiguity policy.

## Evidence Matcher V1 — READY, NOT YET RUN ON MINI256

Authority:

- `experiments/iris_controlled_v1/iris_evidence_matcher_v1.py`
- `experiments/iris_controlled_v1/eval_iris_evidence_matcher_v1.py`
- `experiments/iris_controlled_v1/MINI256_AND_MATCHER_V1_PREREG_20260823.md`

D3 is **not** included. D3 remains reserve-only if a reproducible hard tail survives Matcher V1.

Matcher V1 uses only legal inference evidence:

1. controlled row-corridor candidate domain;
2. Z_coarse top8 retention;
3. FIT-only calibrated rank fusion of Z_coarse + Z_fine + uncertainty-normalized P distance;
4. reciprocal top4 consistency;
5. 3+-view cycle consistency in **image coordinates**;
6. consistency-first rerank;
7. FIT-calibrated singleton margin targeting >=99% precision;
8. singleton / adaptive top4 / adaptive top8 ambiguity output.

GT physical track IDs are used only by evaluator scoring, never matcher decisions.

Drive SHA authorities:

- `iris_evidence_matcher_v1.py` = `4c5b8c80ead2815e1bc9c512f2a5ec8669b5ee48ce22031ae46d083a577edc71`
- `eval_iris_evidence_matcher_v1.py` = `afb51ef36719512c46093201da94678ed0670b17032cf657482b299f9c788f91`
- `run_iris_controlled_v1_mini256_v1.py` = `435b393742c7136d0545b5f398f4bcf4d06e2249d7558a47cf87c048dbad8e2f`
- Mini256 prereg = `816259094cb3c12bd238450d496f925cfe78682a05121407ee81ab54a11f8e6e`

## NEXT EXECUTABLE STEP — MINI256

Do **not** jump to full 3248 training.

Mini256 is preregistered:

- 224 FIT / 32 TUNE;
- deterministic hash selection;
- previous ceiling64 retained as subset;
- 12 epochs;
- corrected `train_iris_controlled_v1_v1_1.py`;
- both cel-clean and ink-cel Matcher V1 evaluation;
- CAL/DEV/EXTERNAL closed.

Notebook in Drive package:

`RealSaS_IRIS_Controlled_V1_Mini256_MatcherV1.ipynb`

Direct command:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python run_iris_controlled_v1_mini256_v1.py --workers 6 --epochs 12
```

It reuses valid existing local cache entries in `/content/IRIS_CONTROLLED_V1_FAST_CACHE` while the Colab runtime lives.

## Mini256 decision policy — frozen before run

GREEN requires:

- >=50% TUNE P error reduction from random init;
- >=50% TUNE N error reduction;
- Matcher V1 final top4 >= .90;
- final top8 >= .97;
- adaptive-set coverage >= .98;
- calibrated confident-singleton precision >= .97 at coverage >= .10.

AMBER: learning exists but Matcher V1 misses GREEN without RED. Do not scale; ablate matcher components.

RED if P or N reduction < .30, final top8 < .90, or adaptive-set coverage < .90.

**GREEN authorizes 1024 intermediate, not immediate full 3248.**

## Trainer lock

Current trainer authority: `train_iris_controlled_v1_v1_1.py`, SHA-256 `961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3`.

Warmup remains P/N/U-only, but TUNE checkpoint selection uses the same full objective every epoch.

## Authorization state

`MATCHER_V1 = READY`

`MINI256 = AUTHORIZED`

`1024_INTERMEDIATE = NOT_AUTHORIZED_UNTIL_MINI256_GREEN`

`FULL_3248_PRODUCTION = NOT_AUTHORIZED`

`CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

## VERY IMPORTANT M4 audit

Authority:

`experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

Mandatory downstream question remains open: can richer observable/set-valued evidence produce a RigAnything-like 2.5D equivalent substrate sufficient for Geppetto and functional rigging? E0–E5 remains required before product substrate closure.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> only then information limit`
