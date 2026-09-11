# RealSaS — Arachne V7-Native A1 FIT1 Authorization

**Date:** 2026-09-11  
**Status:** `A1_MAGE_FIT1_OPTIMIZER_AUTHORIZED__EXACT_SOURCE_AND_PREREG_SEALED`  
**Branch:** `exp/arachne-a1-v7-native-fit1-20260911`  
**Product PASS:** not authorized / not claimed.  
**Unseen/generalization PASS:** not authorized / not claimed.

## Frozen parent

A0 K4 is closed and immutable for this treatment:

- model SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`
- A1 supervision/evaluation bank SHA-256: `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`
- closure result SHA-256: `14043d1f2b638d576e86e25568ffa80e931f84807fafe0135f3f328e027abd6c`
- K4 closure prereg SHA-256: `3ffd304691f836926f2e7787d22b6608ec1a249206f6c2535177f2fd9f4fc857`

The K4 decoder/interface is frozen. Ordered latent alignment remains forbidden as semantic authority because FP32 full-set token permutation invariance passed (`5.7220458984375e-06 <= 1e-05`).

## Exact A1 treatment

Primary preregistration authority for execution:

- `canonical/ARACHNE_A1_V7_NATIVE_FIT1_PREREG_FIX1_20260911.json`
- SHA-256: `631ee85bf2e5822f2cc514e76828652fc9341c05f53460a2459c354a6798fd08`

The earlier prereg SHA `faea572a7dee8191eaaa67da90224d2fd381e9991176b26ba82de267b731aed5` was superseded **before any A1 optimizer construction** only to preserve a deterministic canonical-view yaw code beside the typed per-view evidence. Predictor architecture, objective, optimizer, gates and frozen A0 interface were not changed by that supersession.

Bound model contract:

- architecture: `RealSaS.Arachne.A1.RichQualifiedSurfaceSkeleton.v3`
- architecture config hash: `2ad623648cbe94d5684416f2ddac779dc343661ca9102d526998b37c62865621`
- trainable predictor parameters: `54,031,920`
- field tokens: `4`
- latent width: `512`
- non-autoregressive
- no fixed bone vocabulary / learned joint-ID embedding
- no hard nearest-joint mask
- frozen V7 decoder

Bound objective:

- loss config hash: `c7657fe161eaacfea28e38a1645c744f7ab7bd8c49aceabe1eacfd5f74a78458`
- scalar BCE `1.0`
- scalar MSE `0.1`
- scalar Dice `1.0`
- normalized coupled row-L1 `1.0`
- ordered latent alignment `0.0`
- deformation training loss `0.0`

Teacher W is objective/evaluation target only and is forbidden from predictor input.

## Rich-boundary closure of audit P0

The treatment does not use the historical 20D/8D A1 summaries as the product information boundary. It preserves/uses typed legal evidence including exact GSA topology, per-view support/raster evidence, exact accepted qualified tree, sparse joint->surface mechanical-anchor identity, and deterministic 10D point↔joint/parent-segment geometry.

Current IRIS legacy `log_uncertainty`, saturated Geppetto salience magnitude, hidden/source teacher mesh geometry, source bone names, character identity and raw rejected parent/root alternatives are excluded from baseline neural inputs.

## FIT1 execution contract

- A100 + BF16 forward, FP32 master weights/loss
- AdamW, LR `1e-4`, WD `1e-4`
- warmup `256`, cosine floor `0.1`
- max `8192` steps
- minimum closure step `2048`
- check every `256`
- training query mix: `192 GSA950 supervised + 192 Dense8K product-surface` rows
- GSA row-L1 p95 <= `0.05`
- GSA deformation continuity diagnostic <= `0.05`
- stable observations = `3`
- Compiler `qualify_skin()` required
- disjoint-surface holdout remains diagnostic only

The current synthetic translation deformation probe is retained only for A0-continuity reporting. It is not a training loss and is not a physical deformation proof.

## Authorization

All frozen-parent, source, conditioning, objective and evaluation contracts required for the Mage A1 FIT1 treatment are now sealed.

`A1_MAGE_FIT1_OPTIMIZER_AUTHORIZED = TRUE`

Any architecture/objective/data-boundary change after this record requires a new preregistration transaction.
