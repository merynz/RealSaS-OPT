# RealSaS N1D — Candidate-Conditioned Amplitude P0.5 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DIAGNOSTIC__NO_BACKBONE_TRAINING`  
**Parent P0:** `FAIL__PACTV_SUPPORTED__DIRECT_LOGAMP_NOT_SUPPORTED`  
**Frozen H authority:** F16 Bounded H Research Contract V1

## Question

P0 falsified family-robust **direct/global truth-amplitude regression**, but the architecture does not require `log_amp` to create free displacement. Final XYZ is constrained to frozen `H_i`.

P0.5 asks:

> Is amplitude evidence family-robust when expressed **relative to each carrier's feasible F16 H amplitude distribution**, so it can score candidate magnitudes rather than regress a free absolute target?

## Panel / split

Same broad e00 truth-open families:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Strict leave-one-family-out. Frozen checkpoint and F16 H; no H retuning.

Evaluate only reliable truth-active carriers whose F16 H contains the truth endpoint under the existing primary-2x containment criterion. This isolates amplitude scoring from cases where geometry itself lacks the target.

## Candidate-conditioned target

For carrier `i`:

```text
a_h = ||h - P_A(i)||          for h in frozen H_i
h*  = argmin_h ||h - truth_P_B(i)||   # evaluator-only oracle candidate
```

Define log candidate amplitudes:

```text
l_h = log(eps + a_h)
median_i = median(l_h)
IQR_i = max(q75(l_h)-q25(l_h), eps)
z_h = (l_h - median_i) / IQR_i
z*_i = z_{h*}
```

Also report empirical oracle amplitude quantile `q*_i` within `a_h` as a secondary diagnostic.

`z*` is not a free displacement. It is a coordinate **inside the already-feasible H amplitude distribution**.

## Frozen evidence arms

### R0 — raw current, H-normalized

```text
a_cur = ||consensus(delta_point_map_srcA)||
z_cur = (log(eps+a_cur)-median_i)/IQR_i
```

Use `z_cur` directly as the amplitude score.

### R1 — calibrated current-relative scalar

Strict LOFO Ridge (`alpha=10`) maps scalar `z_cur -> z*` using seven families only.

### R2 — typed scalar + H distribution

Use the P0 B1 typed scalar observation features, plus observation-only H amplitude summaries:

```text
median(log a_h)
IQR(log a_h)
q10/q25/q75/q90(log a_h)
min/max(log a_h)
H amplitude span
current z_cur
```

Standardized Ridge, fixed `alpha=10`, seven-family fit only. No descriptor Z.

## Metrics

On truth-active/F16-contained held-out carriers report per family and pooled:

```text
Spearman(predicted z, oracle z*)
pairwise ordering accuracy for carrier pairs with >=10% oracle-amplitude separation
median |predicted z - z*|
```

Candidate-conditioned magnitude selection diagnostic:

1. choose candidate magnitude whose `z_h` is closest to predicted z;
2. compare selected amplitude to oracle candidate amplitude;
3. report fraction within 25% amplitude ratio and median absolute log-ratio error.

This does not select final XYZ/direction; it evaluates amplitude energy only.

## Gates

For an arm to support candidate-conditioned `log_amp`:

```text
pooled Spearman >= 0.75
median family Spearman >= 0.65
worst evaluable family Spearman >= 0.45
zero negative family Spearman
mean family pairwise accuracy >= 0.72
worst family pairwise accuracy >= 0.60
pooled candidate amplitude within-25% >= 0.70
worst family within-25% >= 0.55
```

## Decision

1. If R1 passes, prefer R1 unless R2 improves worst-family Spearman by >=.05 and worst-family within-25% recall by >=.05 with no new tail failure.
2. If R1 fails but R2 passes, support R2 typed+H input contract.
3. If only R0 passes, retain direct frozen current H-relative amplitude evidence; do not train a richer log_amp head.
4. If no arm passes, candidate-conditioning does not rescue amplitude. The next work must identify an additional local differential evidence channel rather than train through the failure.

## Boundaries

- `p_active` remains separate and already supported by P0 typed evidence.
- `dir` remains separate.
- no descriptor Z fusion;
- no free XYZ;
- no large/end-to-end training;
- no qualification claim.
