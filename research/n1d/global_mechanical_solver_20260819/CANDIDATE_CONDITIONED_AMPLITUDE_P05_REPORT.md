# RealSaS N1D — Candidate-Conditioned Amplitude P0.5

**Date:** 2026-08-19  
**Verdict:** `FAIL__AMPLITUDE_EVIDENCE_CHANNEL_INSUFFICIENT`  
**Status:** `TRUTH_OPEN_FROZEN_H_DIAGNOSTIC__NOT_TRAINING__NOT_QUALIFICATION`  
**Prereg commit:** `de4ca727956a96db095181f49613db8ff9e7ee24`

## Question

P0 showed that `p_active` is family-disjoint learnable but direct/global `log_amp` regression is not. P0.5 tested the more architecture-aligned hypothesis that amplitude need only be expressed **relative to each carrier's frozen F16 feasible H distribution**.

For each reliable truth-active carrier whose F16 H contains the truth endpoint, the evaluator-only oracle candidate `h*` is the H member nearest truth `P_B`. Candidate amplitudes are normalized inside that carrier's own log-amplitude distribution by median/IQR, producing oracle coordinate `z*`.

The prediction never creates XYZ; it only predicts a coordinate inside frozen feasible H amplitude space.

Evaluated carriers: **264**.

## R0 — raw current H-relative coordinate

```text
pooled Spearman              .16383
median family Spearman       .22568
worst family Spearman       -.21811
negative families             1
mean family pairwise acc      .57927
worst pairwise acc            .36876
pooled amplitude within25%    .18939
worst family within25%        0
```

This fails strongly. H-normalizing raw current amplitude does not make its ordering family-robust.

## R1 — calibrated current-relative scalar

A strict LOFO ridge mapping `z_cur -> z*` improves local candidate amplitude matching but not ordering:

```text
pooled Spearman              .11780
median family Spearman       .22568
worst family Spearman       -.21811
pooled within25%             .45076
worst family within25%       .27273
```

Calibration can move predictions toward the typical H amplitude region, but the carrier-to-carrier amplitude-order signal remains wrong on the tail.

## R2 — typed scalar evidence + H amplitude summaries

```text
pooled Spearman              .33528
median family Spearman       .39128
worst family Spearman        .05475
negative families             0
mean family pairwise acc      .54836
worst pairwise acc            .44319
pooled within25%             .46970
worst family within25%       .39216
```

Per-family Spearman:

```text
9908   .24583
11032  .45983
12772  .60909
13203  .38336
14404  .39919
14702  .25920
14758  .64338
15290  .05475
```

R2 removes the negative family but remains far below the preregistered amplitude-ranking gates. `15290` remains effectively unranked.

## Scientific conclusion

Candidate-conditioning **does not rescue** the current amplitude evidence.

This is a sharper conclusion than P0:

> The failure is not merely that we asked the head to regress an absolute/global truth amplitude. Even when the target is expressed entirely inside the carrier's already-feasible frozen H amplitude distribution, current scalar/typed evidence does not provide family-robust magnitude ordering.

Therefore we should not continue target reformulation or train a `log_amp` head through this failure.

At the same time, this does **not** revoke the supported geometry representation or `p_active` result:

- F16 H remains a strong bounded XYZ substrate;
- `p_active` typed evidence remains supported;
- direction remains separate;
- the missing component is specifically a **local differential amplitude evidence channel**.

## Next architecture hypothesis

The most promising missing evidence is candidate-specific multiview motion agreement.

For each frozen H candidate `h` and usable view `v`:

```text
predicted_2d_motion(i,h,v) = project(h,v) - project(P_A(i),v)
```

Compare this candidate-specific motion against observation-native A->B motion evidence at the carrier:

```text
DIS / optical-flow displacement
transport_offset_srcA
or another frozen paired-view local motion field
```

This creates an energy **per candidate** rather than compressing the entire local motion observation into one carrier scalar:

```text
E_amp(i,h) = robust multiview magnitude residual
E_dir(i,h) = robust multiview direction residual
```

The first causal test must keep amplitude and direction residuals separate.

If candidate-specific amplitude agreement ranks truth-near H candidates across the `13203/15290` tails, `log_amp` should be redefined as a candidate energy / local differential compatibility factor rather than a standalone predicted scalar.

If even direct observation-native candidate motion agreement fails, then the paired raster input itself lacks a necessary local differential cue for those cases or our current front-door motion observation is inadequate.

## Authorization boundary

- `p_active`: supported for later calibrated small head.
- standalone/direct/H-relative `log_amp`: **not authorized**.
- combined factor-head training: **not authorized**.
- naive descriptor Z fusion: remains forbidden.
- next work: deterministic candidate-specific local motion energy diagnostic, no backbone training.
- large/end-to-end training remains forbidden.

## Reproducibility

Exact local result SHA-256:
`27ccdea4e42ea535d7a7eb3622ef5dc280de53968cc8770df628652fe1ec3402`

Exact local source SHA-256:
`8988a41bf077b0d5aa8f53f43336072f4ff90b7390965ca6f8faf576e199462c`
