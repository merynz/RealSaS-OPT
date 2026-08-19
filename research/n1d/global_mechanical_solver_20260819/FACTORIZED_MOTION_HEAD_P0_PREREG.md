# RealSaS N1D — Factorized Motion Head P0 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_SMALL_FROZEN_EVIDENCE_PROBE__NOT_END_TO_END_TRAINING`  
**Bounded H authority:** `FREEZE_F16_FOR_RESEARCH__PRODUCT_COMPRESSION_DEFERRED`

## Purpose

Test whether separate motion-state and conditional-amplitude heads are family-disjoint learnable from the already-frozen raster evidence **before** modifying or training the N1D backbone.

This is a calibration/probe experiment, not final neural architecture training.

## Frozen panel

Broad e00 open-development families:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Use the same frozen checkpoint, Problem-A carriers, view visibility and truth mapping as the Representation Sufficiency / F16 H work. Strict leave-one-family-out evaluation: seven families fit/calibrate, one family held out.

Truth is used only as training target/evaluation for the small head. F16 H remains frozen and is not retuned.

## Carrier targets

For reliable mapped carrier `i`:

```text
truth_delta_i = truth_P_B(i) - truth_P_A(i)
truth_amp_i   = ||truth_delta_i||
active_i      = truth_amp_i > 0.005
obs_scale_i   = local neighbor scale of observation-derived P_A carriers
m_i           = log(eps + truth_amp_i / obs_scale_i)
```

`obs_scale_i` is inference-time geometry and does not use truth.

`log_amp` is evaluated conditionally on truth-active carriers. `dir` is explicitly outside P0 and remains a separate factor.

## Frozen observation features

### B0 — raw current baseline

```text
log(eps + ||consensus(delta_point_map_srcA)||)
```

This is the existing strongest simple motion-state/amplitude scalar.

### B1 — typed scalar evidence

Observation-only carrier features sampled at P_A:

```text
current delta consensus xyz
current amplitude
per-view current amplitude mean/std/max/median
view-vector agreement = ||consensus_delta|| / mean_view_amp
transport_gate mean/std/max
transport_entropy mean/std/max
transport_peak mean/std/max
transport_edge_mass mean/std/max
descriptor_log_sigma mean/std
point/log_sigma_A mean/std
visibility probability/support summary
usable view count
observation P_A local scale
```

No free XYZ output is trained.

### B2 — typed scalar + descriptor Z

B1 plus the frozen 64-D multiview descriptor consensus `Z` at P_A. Linear/regularized heads only.

The purpose of B2 is to test whether Z adds generalizable factor information or recreates the previously observed family-specific fusion failure.

## Models

No backbone gradients.

### p_active

For B0/B1/B2:

- standardized logistic regression;
- L2 regularization fixed at `C=0.3`;
- class weighting `balanced`;
- fit on seven families only;
- decision threshold chosen on the seven-family training fold by maximizing balanced accuracy over the training predictions; the held-out family is untouched.

### log_amp

For B0, the raw scalar itself is the ranking score.

For B1/B2:

- standardized Ridge regression;
- fixed `alpha=10`;
- fit on truth-active carriers from the seven training families;
- target `m_i` above;
- held-out family untouched.

No hidden-layer search, family-specific tuning or hyperparameter sweep is allowed in P0.

## Metrics

### p_active

Report pooled and per-family:

```text
AUROC
balanced accuracy at train-selected threshold
Brier score
```

### log_amp, truth-active only

Report pooled and per-family:

```text
Spearman rank correlation
pairwise ordering accuracy for pairs with >=10% truth-amplitude separation
median absolute error in normalized log-amplitude target
```

## Promotion gates

A candidate arm may authorize a later learned small head only if all hold:

### p_active

```text
pooled AUROC >= 0.93
worst evaluable family AUROC >= 0.88
no evaluable family AUROC < 0.85
pooled balanced accuracy >= 0.82
worst-family balanced accuracy >= 0.70
```

### log_amp

```text
pooled active Spearman >= 0.75
median family Spearman >= 0.65
worst evaluable family Spearman >= 0.45
zero negative family Spearman
pooled pairwise ordering accuracy >= 0.72
worst evaluable family pairwise accuracy >= 0.60
```

### tail/fusion rule

B1/B2 must not create a held-out family collapse analogous to the historical `12832` multifeature fusion failure. Any negative family Spearman or p_active AUROC <0.85 is an immediate RED for that arm even if pooled metrics pass.

## Decision

1. If B1 passes both p_active and log_amp gates, prefer B1 unless B2 improves worst-family p_active AUROC by >=.02 **and** worst-family log_amp Spearman by >=.05 without any new tail failure.
2. If B1 fails but B2 passes, B2 is the P0-supported input contract.
3. If only B0 passes, do not train a richer head; current scalar evidence remains the safer factor input and the feature contract must be redesigned.
4. If no arm passes, factor-head training is not authorized and representation/factor evidence must be revisited.

## Boundary after P0

P0 passing does **not** authorize large training. It only authorizes a small learned/calibrated `p_active + log_amp` module with the P0-supported input feature contract, still constrained by frozen F16 H and still evaluated family/episode-disjoint before candidate scoring.
