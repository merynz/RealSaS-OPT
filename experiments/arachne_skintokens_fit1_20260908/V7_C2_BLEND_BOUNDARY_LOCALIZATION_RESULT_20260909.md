# Arachne Mage A0 FIT1 — V7-C2 Blend-Boundary Localization Result

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Preregister commit: `f56957723b6c2d9f67e43f17ddc0a4f6cf92084f`
Diagnostic source commit: `7fcc750dad1e79ef9d95d7ff1f5a5454c269fa99`
Parent top-4 result commit: `335139fc8a29392e5dcb63edde9a47110b786f8d`
C2 treatment model SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
Upstream SkinTokens audited commit: `273b691d35989d71cd17ff2895fdc735097b92d1`

## Scope / limitation

Frozen localization only. No training, optimizer/backward, threshold sweep, K sweep, architecture change, or teacher use in product prediction occurred.

The bound Mage conditioning cache does not contain original triangle-face adjacency. Therefore this diagnostic is explicitly a **joint-normalized point-cloud boundary proxy**, not an exact replay of SkinTokens' triangle-face support-neighborhood mask. Its purpose is to test whether the already-observed 55 top-4 support displacements localize to the same geometric/weak-weight regime targeted by upstream boundary-aware dense sampling.

## Parent failure signature

Frozen C2 treatment + upstream top-4 production mapping:
- row-L1 p95: `0.17656562418530558`
- deformation-error ratio: `0.0774342343211174`
- rows where a false joint displaces at least one true support joint: `55`

Displacement rate by teacher support cardinality:
- support 1: `0 / 444 = 0.0`
- support 2: `3 / 149 = 0.020134228187919462`
- support 3: `45 / 330 = 0.13636363636363635`
- support 4: `7 / 11 = 0.6363636363636364`

The error therefore grows sharply with blend cardinality and is absent on pure one-joint rows.

## Missed true influences are weak

Across the 55 missed-true pairs, teacher-weight percentile inside that joint's positive-weight distribution:
- median: `0.06521739130434782`
- q25: `0.028888888888888888`
- q75: `0.10934782608695652`
- fraction <= 0.25: `0.9454545454545454`
- fraction <= 0.50: `1.0`

Thus 94.5% of missed true influences are in the weakest quartile of their own joint's positive-weight field, and every missed true influence is in the lower half.

## Missed true influences are boundary-localized

Point-cloud support-boundary percentile for those same 55 missed-true pairs:
- median: `0.1625`
- q25: `0.05`
- q75: `0.37555555555555553`
- fraction <= 0.25: `0.60`
- fraction <= 0.50: `0.8727272727272727`

Thus the missed true influences are strongly concentrated near the low-distance support-boundary regime under the preregistered proxy.

## False winners are also near their own support

Across 106 false-kept pairs, point-cloud near-support percentile relative to that false joint's true support:
- median: `0.13859910581222057`
- q25: `0.030736332994736354`
- q75: `0.4274356455316143`
- fraction <= 0.25: `0.5849056603773585`
- fraction <= 0.50: `0.7924528301886793`

Local nearest-neighbor distance median used by the proxy: `0.029196940122451814`.

## Preregistered classification

Alignment tier: **`STRONG`**.

The residual is no longer best described as generic inactive leakage. The supported failure signature is:

**`WEAK_TRUE_VS_NEARBY_FALSE_SUPPORT_ORDERING_AT_BLEND_BOUNDARIES`**

The model usually knows the dominant/major support, but at blend boundaries weak true influences compete with false influences that are themselves geometrically near their own support regions. This is exactly the regime that upstream SkinTokens' support-plus-near-support dense surface sampler is designed to expose more densely during reconstruction training.

## Decision

Authorized next treatment: **SkinTokens-style boundary-aware dense sampling**, isolated as the only training-policy change relative to a matched continuation control.

Still unauthorized:
- custom blend-ratio loss;
- architecture/FSQ/token-count change;
- blind 4k continuation;
- extra characters/pretraining;
- silent adoption of top-4 as a weaker FIT1 acceptance gate.

Because exact upstream triangle faces are unavailable in the current cache, the next treatment must be labeled a **point-cloud proxy port**. Upstream source-code defaults `max_distance=0.1` and `rate_distance=0.1` may be used, but they must be described as code defaults, not proven paper-training override values.