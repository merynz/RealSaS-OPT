# SkinFieldCodec A0 — reconstruction-only and optimizer-cooling diagnostics

**Date:** 2026-09-03  
**Scope:** generic preregistered synthetic witnesses only  
**Frozen behavioral authority:** `test_arachne_v2_behavioral_panel_bound_v2.py`  
**Family-specific tuning:** FORBIDDEN / NONE USED

## Question 1 — Is A0 deformation MSE the generic cause of the clean sharp stability failure?

Full-panel causal A/B preserved witness, seed, model initialization, AdamW LR/WD, horizon, evaluation cadence, acceptance thresholds and three-consecutive-PASS rule. The only lane difference was `deformation_weight=1.0` versus `0.0`. Deformation remained an authoritative evaluation metric in both lanes.

Workflow `33758209799`, job `100657705289`:

- `chain_blend_3`: current PASS step `480`; reconstruction-only PASS step `896`.
- `branch_blend_4`: current PASS step `768`; reconstruction-only PASS step `672`.
- `sharp_fork_5`: current FAIL; reconstruction-only also FAIL.
  - current final p95 `0.0448788`, deformation ratio `0.00790533`, stable count `1`;
  - reconstruction-only final p95 `0.0521763`, deformation ratio `0.00776672`, stable count `0`.

The earlier post-PASS branch experiment remains valid as a local optimizer-path observation: from the same step-1280 sharp checkpoint and Adam state, 32 current-objective steps produced p95 `0.0831804`, whereas reconstruction-only produced p95 `0.0495638`. But the full-horizon panel falsifies promotion of that local effect into a generic root-cause claim.

**Decision:**

`A0_DEFORMATION_MSE_IS_THE_GENERIC_SHARP_ROOT_CAUSE = FALSE`

No removal or retuning of the deformation term is authorized.

## Question 2 — Is constant LR overshoot the generic cause?

A second full-panel A/B preserved the full current A0 objective, model, seed, AdamW weight decay, horizon and frozen acceptance. The only difference was optimizer schedule:

- current constant LR `1e-3`;
- generic `CosineAnnealingLR(1e-3 -> 0, T_max=1536)`.

Workflow `33758542932`, job `100658825104`:

- `chain_blend_3`: constant PASS step `480`; cosine PASS step `512`.
- `branch_blend_4`: constant PASS step `768`; cosine PASS step `384`.
- `sharp_fork_5`: both FAIL.

Sharp cosine lane is especially informative. As LR approaches zero, deformation remains excellent while row-p95 converges to a narrow non-PASS floor:

- step `1376`: p95 `0.0508867`;
- step `1408`: `0.0508192`;
- step `1440`: `0.0508573`;
- step `1472`: `0.0505474`;
- step `1504`: `0.0504006`;
- step `1536`, LR `0`: p95 `0.0504013`, deformation ratio `0.00662181`.

**Decision:**

`CONSTANT_LR_OVERSHOOT_IS_THE_GENERIC_SHARP_ROOT_CAUSE = FALSE`

No scheduler source repair is authorized.

## Updated interpretation

The clean sharp seam is no longer best described as stochastic or optimizer oscillation. Codec dropout is `0.0`, deterministic execution is enabled, and cosine cooling removes late step-size motion without crossing the frozen p95 ceiling. The remaining evidence is consistent with a narrow representational/objective floor inside the current Codec encode/decode chain, but does not yet identify whether the first bottleneck is teacher encoding or latent-to-weight decoding.

The next authorized causal split is therefore:

`teacher encoder -> latent -> decoder`

versus

`free per-joint latent -> same decoder`.

This split must use the corrected canonical surface-ID binding and identical downstream objective/evaluation. Historical pre-binding free-latent diagnostics remain contaminated and cannot be reused as evidence.

## Epistemic status

- teacher-W row binding bug: `CONFIRMED / REPAIRED IN V2 HARNESS`;
- exact-truth CE stationarity bug: `CONFIRMED / SOURCE REPAIRED`;
- top-10% tail repair: `FALSIFIED`;
- fixed global temperature: `FALSIFIED`;
- weight decay: `FALSIFIED`;
- deformation-MSE removal as generic solution: `FALSIFIED`;
- cosine optimizer cooling as generic solution: `FALSIFIED`;
- encoder-vs-decoder bottleneck: `UNKNOWN / NEXT CAUSAL GATE`.

Global refreeze and formal Family-1 selection remain blocked.
