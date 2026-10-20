# IRIS H-GEO-P3-SCALE-XFER — Decision Ledger

Date: 2026-10-20

## Trigger

H-GEO-P3-EFREQ completed with zero optimizer/probe steps and classification `UNRESOLVED`. The locked EFREQ dominance gates did not permit choosing P4, but the measured result showed three unresolved patterns that require stage localization before any new training:

- P3 H_geo is smoother than P2 H at fine perturbation scales;
- endpoint-tail projected surface footprints are smaller than controls;
- endpoint tails exhibit more non-adjacent surface competition.

Authority: `docs/research/IRIS_H_GEO_P3_EFREQ_RESULT_20261020.json`.

## Question

> At which stage do endpoint-relevant fine spatial distinctions first become inadequate or ambiguous: frozen view tokens, P3 scene/carrier transfer, or the single closest-surface target itself?

## Experimental law

This is a **zero-optimizer diagnostic**.

- Reuse the EFREQ query CSV byte-exactly. Tail/control membership is immutable.
- Reuse immutable V8, sealed P2, sealed P3, exact P1 triangle dataset, prepared shards and authored teacher meshes.
- No carrier/probe optimization.
- No checkpoint write.
- No architecture mutation.
- No SURF-A2 or TESSA authorization.

## Measurements

### A. Frozen view-token evidence

For each exact EFREQ query, pair the exact nearest surface point with the closest non-adjacent competing surface point. Project both into the eight normalized cameras and sample the existing 37×37×384 frozen image-token grid with bilinear `align_corners=False` semantics.

Record:

- both-in-frame view count;
- views in which the two surfaces fall in different token cells;
- same-token-cell fraction;
- median / maximum token cosine distance;
- projected separation in token-cell units.

This is a frozen token evidence diagnostic, not a claim of true occlusion visibility.

### B. Exact multi-surface target structure

From authored teacher triangles, find the nearest non-adjacent competing surfaces and record:

- competitor distance margin over the exact nearest target;
- competitor normal angle up to sign;
- whether another surface lies within 1/256, 1/128 or 1/64 IRIS of the nearest distance.

### C. Carrier transfer curve

Use deterministic ± perturbations at:

`1/1024, 1/512, 1/256, 1/128, 1/64, 1/32` IRIS.

Measure per-subject channel-whitened RMS response at:

1. fresh-court P2 H baseline;
2. P3 sampled 64² base planes;
3. P3 sampled bilinear 128² planes;
4. P3 sampled learned-refined 128² planes;
5. P3 geometry-field H;
6. P3 continuous-position residual;
7. P3 final H_geo.

The first stage that materially loses a distinction already present upstream is the candidate bottleneck.

## Locked classification

- `TARGET_MULTISURFACE_LIMITED`
- `VIEW_INPUT_RESOLUTION_LIMITED`
- `CARRIER_TRANSFER_LIMITED`
- `MIXED`
- `UNRESOLVED`

Thresholds are in `canonical/H_GEO_P3_SCALE_XFER_PREREGISTRATION_V1.json` and may not be moved after runtime.

## Consequence law

No classification directly authorizes training.

- target-only → preregister multi-surface / multi-hypothesis target P4;
- view-only → preregister multiscale/higher-spatial-resolution view-evidence P4;
- carrier-only → preregister evidence-preserving geometry-carrier P4;
- mixed → preregister the minimal joint P4 containing only identified failing stages;
- unresolved → expand zero-optimizer localization only.

TESSA remains blocked until a genuinely qualified direct geometry A-arm exists.
