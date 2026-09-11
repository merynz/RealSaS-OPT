# RealSaS — Arachne V7-Native A1 FIT1 Authorization

**Date:** 2026-09-11  
**Status:** `A1_V4_FIX1_MAGE_FIT1_OPTIMIZER_AUTHORIZED__PRE_NOTEBOOK_AUDIT_CLOSED`  
**Branch:** `exp/arachne-a1-v7-native-fit1-20260911`  
**Product PASS:** not authorized / not claimed.  
**Unseen/generalization PASS:** not authorized / not claimed.

## Frozen A0 parent

- K4 model SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`
- A1 supervision/evaluation bank SHA-256: `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`
- closure result SHA-256: `14043d1f2b638d576e86e25568ffa80e931f84807fafe0135f3f328e027abd6c`
- closure prereg SHA-256: `3ffd304691f836926f2e7787d22b6608ec1a249206f6c2535177f2fd9f4fc857`
- field tokens: `4`
- latent channels: `512`
- A0 GSA final p95/deformation: `0.04491063521144626 / 0.018186409026384354`
- stable-last-3: PASS
- FP32 K4 full-set permutation delta: `5.7220458984375e-06 <= 1e-05`

The frozen decoder therefore consumes an unordered K4 set. Ordered latent alignment is forbidden as semantic authority.

## Supersession

The earlier V3 A1 authorization is **superseded before any A1 optimizer construction**.

V3 was a useful implementation draft, but the final predictor audit found remaining gaps that were important enough to fix before the first A1 treatment: predictor capacity, one-way rather than bidirectional fusion, absolute view-slot leakage through the flat surface input, pair-mask non-consumption, missing hard-tail/blend objective terms, and no direct physically articulated deformation gradient.

Primary audit closure:

- `canonical/ARACHNE_A1_PREDICTOR_FINAL_AUDIT_V2_20260911.md`

Primary execution preregistration:

- `canonical/ARACHNE_A1_V7_NATIVE_FIT1_PREREG_V4_FIX1_20260911.json`

Execution entry point:

- `experiments/arachne_a1_v4_fit1/run_arachne_a1_v4_fit1_fix1.py`

## Final A1 predictor contract

Architecture: `RealSaS.Arachne.A1.RichQualifiedBidirectional.v4`

- trainable predictor parameters: **138,053,153**
- model width: `640`
- attention heads: `10` (`64D/head`)
- exact-edge surface message layers: `4`
- global surface transformer layers: `6`
- qualified-tree message layers: `4`
- joint-field token self layers: `6`
- dense bidirectional surface↔joint-field fusion rounds: `4`
- J×K on Mage: `22 × 4 = 88` field tokens
- non-autoregressive
- no fixed bone vocabulary
- no learned semantic joint-ID embedding
- no hard nearest-joint mask
- exact pair legality mask is honored
- frozen K4 V7 decoder

The capacity choice is a deliberate middle point in a same-topology static sweep: `88,451,201` params at width 512, `138,053,153` at width 640, and `198,650,817` at width 768. It removes the tiny-predictor confound without claiming globally optimal capacity.

## Rich legal conditioning

Predictor inputs preserve/use:

- compact surface P/N/normal validity;
- exact 8-view support;
- exact per-view raster XY + validity;
- observed/completed state;
- exact GSA topology + numeric edge metadata;
- qualified joint positions;
- exact accepted parent tree/root/deform-root;
- exact sparse joint→surface support-anchor identity;
- deterministic 10D point↔joint/parent-segment geometry;
- exact pair legality mask;
- camera-bound yaw Fourier sidecar.

View-specific evidence is processed only through a shared view encoder and permutation-invariant pooling. It is no longer flattened into an absolute view-slot vector.

Forbidden predictor evidence remains: teacher W, teacher A0 latent, hidden/source teacher mesh, source bone/component names, family/character identity, provenance hashes, rejected Geppetto root/parent alternatives, current uncalibrated IRIS `log_uncertainty`, saturated Geppetto salience, and frozen codec condition tokens as predictor evidence.

## Final objective

Loss config: `RealSaS.Arachne.A1.BehaviorHardTailArticulatedLoss.v4`

- scalar BCE: `1.0`
- scalar MSE: `0.1`
- scalar Dice: `1.0`
- normalized coupled row-L1: `1.0`
- hard-tail CVaR10 row-L1: `0.5`
- blend-boundary row-L1: `0.5`
- articulated deformation consequence: `0.25`
- ordered latent alignment: `0.0`

Teacher W is objective/evaluation only.

The new deformation probe is deterministic, parent-relative and based only on qualified joint geometry/tree. It has an explicit joint-permutation-equivariance preflight. The historical synthetic translation probe remains only an A0 continuity metric.

## Mandatory preflight before optimizer construction

- exact A0 checkpoint SHA;
- exact supervision bank SHA;
- surface/skeleton lineage;
- `950` GSA nodes / `2813` edges / `22` joints;
- exact condition query binding and codec frame parity;
- exact V4 parameter count;
- surface permutation equivariance;
- joint permutation equivariance;
- view/camera-binding permutation equivariance;
- support-anchor remapping consistency through those tests;
- articulated-probe joint permutation equivariance;
- predictor-input teacher/condition-token firewall;
- A100/BF16 and minimum VRAM gate.

Optimizer creation occurs only after all of these checks.

## FIT1 execution contract

- A100 + BF16 forward, FP32 master weights/loss
- AdamW, LR `5e-5`, WD `1e-4`
- warmup `512`, cosine floor `0.1`
- maximum `12,288` steps
- minimum closure step `2,048`
- evaluation every `256`
- train rows/step: `192 GSA950 + 192 Dense8K`
- GSA row-L1 p95 <= `0.05`
- historical continuity deformation ratio <= `0.05`
- Compiler qualified surface rows = `950`
- Compiler aggregate correction L1 <= `1e-4`
- stable observations = `3`
- disjoint-surface holdout remains diagnostic only

The articulated deformation metric is trained and reported in this first V4 treatment but does not receive a post-hoc numerical PASS threshold.

## Authorization

`ARACHNE_A1_PREDICTOR_PRE_NOTEBOOK_AUDIT = CLOSED`

`A1_V3_EXECUTION_AUTHORITY = SUPERSEDED_PRE_OPTIMIZER`

`A1_V4_FIX1_MAGE_FIT1_OPTIMIZER_AUTHORIZED = TRUE`

`NOTEBOOK_GENERATION = AUTHORIZED`

Any architecture, feature-boundary, objective, probe, frozen-parent or acceptance-gate change after this record requires a new preregistration transaction.
