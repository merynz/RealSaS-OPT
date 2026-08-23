# RealSaS-OPT — Current State

**Date:** 2026-08-23  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `IRIS_CONTROLLED_V1_REPRESENTATION_PASS__OPEN_PILOT_PASS__PRODUCTION_OPEN_TRAINING_AUTHORIZED__SEALED_CLOSED`

## Read this first

This file is the single continuation authority. If a new chat says only `devam et`, continue from the exact next action below without reconstructing history from the user.

## Scientific correction

**Observable evidence was not pruned; the problem was pruned.**

IRIS target:

`8 controlled neutral views -> persistent observable surface geometry + uncertainty`

Legal evidence: P, geometric N, support/visibility, uncertainty/risk, coarse/fine persistence, reciprocal/cycle consistency, set-valued ambiguity, provenance.

Forbidden IRIS target authority: hidden joint/owner IDs, parent graph, skin weights, pose-B mechanics, GFDR hidden mechanics.

## Corpus

Canonical selected: 3993.

Preserved exclusions:

- 56 repair-pending;
- 6 active non-Basis shape-key;
- 1 all-8 blank.

Controlled V1 usable: **3930**.

Frozen split:

- FIT 2935
- TUNE 313
- CAL 246 — SEALED
- DEV 270 — SEALED
- EXTERNAL_HOLDOUT 166 — SEALED

Open FIT+TUNE = **3248**. No resplit.

## Base package authority

Drive:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1`

Folder ID: `15Du2plm2vHYe4Mmm-p1-L6emkxYucN8k`

Base `PACKAGE_MANIFEST_V1.json` SHA-256:

`e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098`

CAL/DEV/EXTERNAL remain unopened.

## Representation ceiling — PASS

Authority:

`experiments/iris_controlled_v1/CEILING_PASS_AND_OPEN_PILOT_DECISION_20260823.md`

64 deterministic open assets (51 FIT / 13 TUNE), 65,830 cross-view pairs:

- P exact top1 `0.9976302598`
- P exact top4 `1.0`
- P exact top8 `1.0`
- P exact family top1 p10 `0.9934295619`
- P+N exact top8 `0.9998936655`
- P noise .0025 top4/top8 `1.0 / 1.0`
- P noise .005 + N noise 20° top4 `0.9988151299`
- same heavy-noise top8 `0.9996658059`
- ambiguity within 0.003 `0.0151754519`

Interpretation: observable addressability exists strongly in the controlled setting. P alone slightly beats naive fixed-weight P+N top1; N remains rich evidence but must not be forced into matching with an uncalibrated constant coefficient. High-recall top4/top8 strongly supports set-valued ambiguity rather than forced singleton identity.

## Open learner pilot — PASS

Authority:

`experiments/iris_controlled_v1/OPEN_PILOT_PASS_AND_PRODUCTION_AUTHORIZATION_20260823.md`

64 assets, 51 FIT / 13 TUNE, 8 epochs.

Random-init → trained TUNE:

- P Euclidean `0.6096198788 -> 0.1311205992` = **78.49% reduction**
- N error `0.9678560908 -> 0.1776760645` = **81.64% reduction**
- coarse top8 `0.1243990385 -> 0.9651442308` = **+84.07 pp**
- fine top8 `0.1039663462 -> 0.9375` = **+83.35 pp**
- coarse top4 `0.8957331731`
- fine top4 `0.8527644231`
- final TUNE full objective `0.2976353329`

Diagnostic PASS by large margins. This is strong learner evidence, not sealed/product generalization proof.

## Trainer implementation lock

Old checkpoint selection had a pre-production bug: warmup epochs optimized fewer terms, so raw TUNE total was not comparable across epochs.

Current trainer authority:

`train_iris_controlled_v1_v1_1.py`

Drive SHA-256:

`961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3`

It keeps FIT warmup but evaluates TUNE checkpoint selection with the same full post-warmup objective every epoch.

## Cache engineering

Original serial prepare is deprecated. Fast V2 uses local SSD, bounded parallel Drive reads, vectorized correspondence with exact old-vs-new parity, 512→256 derived RGBA preprocessing, and SHA-sealed cache artifacts.

Full-cache random Drive read throughput is still the operational bottleneck. Existing `/content/IRIS_CONTROLLED_V1_FAST_CACHE` partial assets should be preserved while the current Colab runtime lives; the builder resumes/reuses valid local cache files.

## NEXT EXECUTABLE STEP

Full open-cache completion and production Controlled V1 training are now authorized.

Use the corrected production launcher, **not** the old fast launcher's train mode:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python launch_iris_controlled_v1_production_v1_1.py --mode all --workers 4
```

Production launcher SHA-256:

`b88fb1eaea8889019a6e64e015a2e19c278134794111a1cd5ca6f4a79637039a`

It requires:

- frozen base-package verification;
- recorded 64-asset representation ceiling PASS;
- full local open cache `record_count = 3248` plus cache seal;
- corrected trainer v1.1 SHA;
- CAL/DEV/EXTERNAL closed.

If cache completes but training must be resumed later:

```bash
python launch_iris_controlled_v1_production_v1_1.py --mode train --resume
```

## Authorization state

`FULL_OPEN_CACHE = AUTHORIZED`

`CONTROLLED_V1_PRODUCTION_TRAINING = AUTHORIZED_AFTER_FULL_CACHE_SEAL`

`CAL_DEV_EXTERNAL = CLOSED`

`SEALED_EVALUATION = NOT_AUTHORIZED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

## VERY IMPORTANT M4 audit

Authority:

`experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

Frozen interpretation:

- Graph V1 identity path was identity-contracting and falsified;
- Q representation survives;
- QF V1 did not test intended joint A∪B quotient;
- scalar threshold/admission consumer was falsified;
- CORR is causally supported;
- typed signed joint tiny closure passed;
- V2-C did not prove family-disjoint generalization;
- UNKNOWN existed historically;
- current change is problem-pruned, evidence-richer.

Mandatory downstream question remains open: can richer observable/set-valued evidence produce a RigAnything-like 2.5D equivalent substrate sufficient for Geppetto and functional rigging? Run E0–E5 before claiming product substrate closure.

## Research rule

For any future failure classify in order:

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> only then information limit`

Do not infer impossibility from learner failure. Do not infer product sufficiency from oracle/controlled PASS.
