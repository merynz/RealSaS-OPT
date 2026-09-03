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

## Question 3 — Is the teacher-summary encoder the first representation bottleneck?

A clean ID-bound diagnostic used the same decoder, same current objective, same cosine-to-zero schedule, same frozen witnesses and same canonical row binding. The comparison lane replaced the teacher-summary encoder output with a free trainable per-joint latent initialized from the encoder latent.

Result:

- chain: both current and free-latent sustained PASS;
- branch: both current and free-latent sustained PASS;
- `sharp_fork_5`: current final p95 `0.0504013`, free-latent final p95 `0.0558612`; both FAIL.

**Decision:**

`TEACHER_SUMMARY_ENCODER_IS_THE_FIRST_SHARP_BOTTLENECK = FALSE`

Removing encoder compression does not close the sharp seam.

## Question 4 — Does direct point↔joint geometry close the decoder floor?

A zero-initialized residual projection exposed only the existing canonical `point_control_dx/dy/dz/distance` channels to the decoder while preserving the current step-0 function.

Result:

- chain: geometry lane PASS;
- branch: current PASS, geometry lane FAIL with final p95 `0.0638736`;
- sharp: current p95 `0.0504013`, geometry lane p95 `0.0847526`; FAIL.

**Decision:**

`FOUR_CHANNEL_POINT_CONTROL_GEOMETRY_CLOSES_THE_SHARP_SEAM = FALSE`

No 4-channel decoder-conditioning repair is authorized.

## Question 5 — Does the complete canonical V2 pair contract close the decoder floor?

The final authorized feature-level representation diagnostic exposed the complete already-defined ten-channel `PAIR_GEOMETRY_CONTRACT_V2` as one zero-initialized residual block:

`point_control_dx, point_control_dy, point_control_dz, point_control_distance, point_segment_distance, segment_t, segment_length, normal_axis_abs_cos, normal_axis_valid, parent_exists`.

No subset search or witness-conditioned channel tuning was permitted.

Workflow `33760883020`, job `100666623311`:

- chain: current PASS; full pair-contract PASS step `480`, final p95 `0.0395030`;
- branch: current PASS; full pair-contract PASS step `640`, final p95 `0.0405289`;
- sharp: current FAIL at `0.0504013`; full pair-contract also FAIL, final p95 `0.0774042`, deformation ratio `0.0150600`.

**Decision:**

`FULL_CANONICAL_PAIR_CONTRACT_CLOSES_THE_SHARP_SEAM = FALSE`

Feature addition/subset search is now closed. No further pair-geometry channel hunting is authorized from this witness panel.

## Updated interpretation

The clean sharp seam is not explained by stochastic dropout, constant-LR overshoot, deformation-MSE interference, teacher-summary compression, missing direct point↔joint geometry, or omission of the complete canonical V2 pair contract. The surviving evidence points to a narrower failure inside the current latent→pair-logit→row-softmax reconstruction behavior or its optimization geometry, but the first faulty mechanism is not yet identified.

The next authorized diagnostic is therefore row-level anatomy on the cosine-converged current Codec, with no source behavior change:

- canonical surface-row ID and rest position;
- teacher versus predicted simplex;
- signed per-joint mass delta;
- row-L1 and exact p95 interpolation support;
- decoder centered logits versus teacher-equivalent centered log-probabilities;
- global per-joint signed and absolute mass error.

This diagnostic must run identically on all preregistered witnesses. Its purpose is to identify whether the sharp floor is localized to one/two canonical rows, systematic fork-branch leakage, or a broader logit/softmax approximation floor.

## Epistemic status

- teacher-W row binding bug: `CONFIRMED / REPAIRED IN V2 HARNESS`;
- exact-truth CE stationarity bug: `CONFIRMED / SOURCE REPAIRED`;
- top-10% tail repair: `FALSIFIED`;
- fixed global temperature: `FALSIFIED`;
- weight decay: `FALSIFIED`;
- deformation-MSE removal as generic solution: `FALSIFIED`;
- cosine optimizer cooling as generic solution: `FALSIFIED`;
- teacher-summary encoder bottleneck: `FALSIFIED`;
- four-channel point-control geometry repair: `FALSIFIED`;
- full ten-channel canonical pair-contract repair: `FALSIFIED`;
- row-level sharp error mechanism: `UNKNOWN / ACTIVE DIAGNOSTIC`.

Global refreeze and formal Family-1 selection remain blocked.
