# IRIS Controlled V1 — Open Pilot PASS & Production Authorization

**Date:** 2026-08-23  
**Status:** PASS / production open-split training authorized  
**Sealed:** CAL/DEV/EXTERNAL remain closed

## Pilot panel

- 64 total open assets
- FIT 51
- TUNE 13
- 8 epochs
- same 64-asset cache previously used for the representation ceiling

## Random-init → trained TUNE

- P Euclidean error: `0.6096198788 → 0.1311205992` (**78.49% reduction**)
- N error: `0.9678560908 → 0.1776760645` (**81.64% reduction**)
- coarse retrieval top8: `0.1243990385 → 0.9651442308` (**+84.07 percentage points**)
- fine retrieval top8: `0.1039663462 → 0.9375` (**+83.35 percentage points**)
- coarse top4: `0.8957331731`
- fine top4: `0.8527644231`
- coarse top1: `0.4927884615`
- fine top1: `0.46484375`
- final TUNE full objective: `0.2976353329`

Diagnostic PASS criteria were exceeded by large margins.

## Interpretation

The 64-asset controlled diagnostic now supports both prerequisites:

1. **Representation ceiling PASS:** exact legal observable P/(P,N) can address persistent surface loci at very high recall.
2. **Learner pilot PASS:** the current neural core can extract useful P/N and persistence evidence from RGB on held-out open TUNE assets.

This is not product/generalization closure. The TUNE panel is only 13 assets and sealed panels remain unopened. But the result is strong enough to authorize full open FIT+TUNE preparation and the preregistered Controlled V1 training line.

## Production implementation lock

The original trainer is superseded for checkpoint selection by:

`train_iris_controlled_v1_v1_1.py`

Reason: epochs 0–3 optimize the warmup objective while persistence terms enter at epoch 4; comparing raw TUNE total across changing objective definitions could falsely favor a warmup checkpoint. v1.1 evaluates TUNE selection with the same full objective at every epoch.

Production launcher:

`launch_iris_controlled_v1_production_v1_1.py`

SHA-256: `b88fb1eaea8889019a6e64e015a2e19c278134794111a1cd5ca6f4a79637039a`

It requires:

- frozen base package verification;
- saved 64-asset representation ceiling PASS;
- full open cache record_count = 3248;
- corrected trainer v1.1 SHA;
- CAL/DEV/EXTERNAL closed.

## Authorization

`FULL_OPEN_CACHE = AUTHORIZED`

`CONTROLLED_V1_PRODUCTION_TRAINING = AUTHORIZED_AFTER_FULL_CACHE_SEAL`

`SEALED_EVALUATION = NOT_AUTHORIZED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

The separate M4 E0–E5 equivalent-substrate program remains mandatory before claiming Geppetto/product sufficiency.
