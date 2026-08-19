# RealSaS N1D — Factorized Motion Head P0

**Date:** 2026-08-19  
**Verdict:** `FAIL__PACTV_SUPPORTED__DIRECT_LOGAMP_NOT_SUPPORTED`  
**Status:** `TRUTH_OPEN_SMALL_FROZEN_EVIDENCE_PROBE__NOT_END_TO_END_TRAINING`  
**Prereg commit:** `5ad75bbe5434af901f105f1654e1f8e2b074ff33`

## Scope

Frozen N1D backbone, frozen broad e00 8-family panel, strict leave-one-family-out small linear heads. F16 H is frozen and untouched. P0 asks whether `p_active` and direct conditional truth-amplitude regression are family-disjoint learnable from frozen carrier evidence.

## p_active — supported

### Raw current baseline B0

- pooled AUROC: `0.9311`
- worst family AUROC: `0.8915`
- mean family balanced accuracy: `0.8471`
- worst balanced accuracy: `0.7816`

P0 p_active gate: **PASS**.

### Typed scalar B1

- pooled AUROC: **`0.9585`**
- worst family AUROC: **`0.8961`**
- mean family balanced accuracy: **`0.8918`**
- worst balanced accuracy: **`0.7774`**
- Brier: **`0.0725`** vs B0 `0.1062`

P0 p_active gate: **PASS**.

Typed current/transport/uncertainty evidence materially improves motion-state discrimination and calibration without needing descriptor Z.

## Direct log_amp — not supported

### B0 raw current amplitude ranking

- pooled active Spearman: `0.5852`
- median family: `0.6956`
- worst family: `0.2369`
- pairwise mean family accuracy: `0.7581`
- worst pairwise: `0.6024`

Fails the preregistered amplitude tail gates.

### B1 typed scalar regression

- pooled Spearman: `0.5743`
- median family: `0.7525`
- worst family: **`0.1261`**
- pairwise mean family accuracy: `0.7713`
- worst pairwise: **`0.5489`**

Held-out hard amplitude families:

```text
13203  rho = .1261
15290  rho = .1894
```

The typed scalar features improve p_active but do not make direct absolute/normalized amplitude regression family-robust.

## Descriptor Z fusion — RED

B2 (`typed scalar + 64-D Z`) recreates the historical multifeature family-collapse pathology:

- p_active worst AUROC: `.8330` — FAIL
- log_amp `15290` Spearman: **`-.4033`**
- worst pairwise accuracy: `.3261`

Therefore Z must not be added naively to the motion-factor head.

## Scientific interpretation

P0 cleanly separates the two factors:

```text
p_active  -> observation evidence is strong and family-disjoint enough for a calibrated small head
log_amp   -> direct global truth-amplitude regression is not family-robust under the tested evidence contract
```

This does **not** imply that amplitude information is absent from the full representation. The architecture already constrains final XYZ to F16 `H_i`, and earlier candidate-basin experiments showed scalar evidence can improve activity/motion when composed inside feasible geometry.

The next question is therefore narrower and better aligned with the intended authority split:

> Can raster-derived amplitude evidence rank/score **candidate displacement magnitudes inside frozen H_i**, rather than predict an unconstrained absolute truth amplitude?

A candidate-conditioned objective may need only to prefer the correct magnitude region in the feasible set. It should not be required to reconstruct a free scalar target independently of geometry.

## Next required diagnostic

Before any learned log_amp head:

1. for each reliable carrier, form frozen F16 candidate displacement magnitudes `||h - P_A||`;
2. measure where the truth-near / oracle-best H candidate lies in that magnitude distribution;
3. compare frozen current amplitude and typed scalar evidence against the **candidate rank/quantile** target;
4. evaluate strict family-disjoint candidate-pair ordering, not free amplitude regression;
5. if this closes the `13203/15290` tail, redefine log_amp as a candidate-conditioned energy/rank factor and preregister P1;
6. if it does not, the amplitude factor is missing a local differential evidence channel and representation must be extended before training.

## Authorization boundary

- `p_active` feature contract: **supported for a later small calibrated head**.
- direct `log_amp` head: **not authorized**.
- combined factor-head training: **not authorized yet**.
- descriptor Z fusion into motion factors: **forbidden by P0 evidence**.
- large/end-to-end training remains forbidden.

## Reproducibility

Exact local result SHA-256:
`b5eaf41ec2f1a0ba248061773a8438106e45585ec683d8d649800d0878e462ca`

Exact local source SHA-256:
`a3a952373e5eb4b616c40270d78336e7d6638b2ed5b22f35985ff5b2588df777`
