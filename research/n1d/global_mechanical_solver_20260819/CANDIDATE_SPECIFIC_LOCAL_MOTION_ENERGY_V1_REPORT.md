# RealSaS N1D — Candidate-Specific Local Motion Energy V1

**Date:** 2026-08-19  
**Verdict:** `FAIL__AMPLITUDE_ONLY_LOCAL_MOTION_COMPATIBILITY_NOT_FAMILY_ROBUST`  
**Status:** `TRUTH_OPEN_DETERMINISTIC_DIAGNOSTIC__NOT_TRAINING__NOT_QUALIFICATION`  
**Prereg commit:** `6af5012a37673b5e1826b3d84304463927ff6e23`  
**Implementation freeze:** `7a4d8ce8f4147f90cb054d079bd2c07be45df5b3`

## Purpose

P0.5 showed that carrier-level scalar evidence fails even when amplitude is expressed relative to frozen H. V1 therefore preserved local per-view evidence and scored every frozen F16 candidate directly.

For candidate `h` and usable view `v`:

```text
d_h = project(h,v) - project(P_A,v)
```

Magnitude-only residual:

```text
|log(eps + ||d_h||) - log(eps + ||d_obs||)|
```

was aggregated by median across views.

Observation fields:

- raw raster DIS A->B flow;
- frozen learned `transport_offset_srcA`, converted from 32x32 f4 cell units to native pixels;
- frozen per-view `delta_point_map_srcA`, projected back to 2D;
- deterministic robust fusion of the three amplitude energies.

Evaluation used the same `264` reliable truth-active F16-contained carriers as P0.5.

## Aggregate results

### Raw DIS amplitude compatibility

```text
within25                    .40152
within50                    .60606
worst family within25       .18182
worst family within50       .36364
mean family pair ordering   .70354
worst pair ordering         .57295
median abs log ratio        .30195
13203 within25              .35135
15290 within25              .49020
```

DIS carries useful pairwise ordering signal but does not localize the correct feasible magnitude region reliably enough.

### Learned transport amplitude compatibility

```text
within25                    .34091
within50                    .56439
worst family within25       .18182
mean family pair ordering   .68526
13203 within25              .27027
15290 within25              .45098
```

Fails.

### Per-view projected delta_point_map amplitude

```text
within25                    .29167
within50                    .47348
worst family within25       0
mean family pair ordering   .70257
13203 within25              .27027
15290 within25              .07843
```

The old differential 3D field can order some candidate pairs but is severely family-dependent as an amplitude selector.

### Robust fusion

```text
within25                    .34848
within50                    .58333
worst family within25       .09091
mean family pair ordering   .71342
worst pair ordering         .60149
13203 within25              .24324
15290 within25              .27451
```

Fusion improves pairwise ordering slightly but worsens exact amplitude-region selection on the hard tails.

## Scientific interpretation

No amplitude-only arm passes.

However the failure pattern is informative rather than a simple absence-of-signal result:

- DIS and robust fusion achieve mean family candidate-pair ordering around `.70-.71`;
- several individual families show substantially stronger ordering (e.g. DIS on `15290` ~`.827` and delta3D on `11032` ~`.838` in the per-family records);
- nevertheless the selected candidate magnitude is usually not close enough to the oracle-near H candidate magnitude.

This suggests a structural problem with **forcing candidate compatibility to factor into direction-independent amplitude**.

Under orthographic projection, the observed 2D displacement magnitude depends jointly on the 3D displacement magnitude **and its direction relative to the camera**. Once `h` is already a concrete 3D candidate, discarding vector direction before scoring throws away information that can distinguish candidate magnitude/direction combinations.

Therefore V1 does not yet justify adding a new input channel. It first falsifies the stronger assumption:

> candidate scoring can use an amplitude-only compatibility energy independent of direction.

## Next causal test

Keep all observation fields and F16 H frozen, but test **full 2D vector compatibility** per candidate:

```text
E_vec(i,h) = robust median_v distance(
    project(h,v)-project(P_A,v),
    observed_A_to_B_motion(v)
)
```

Test DIS, transport and delta3D separately and deterministic fusion.

This test must evaluate final candidate endpoint quality, not call the result `log_amp`. If full-vector compatibility succeeds while amplitude-only fails, architecture should change from independent scalar `log_amp + dir` scoring to:

```text
p_active + candidate-conditioned vector/differential compatibility
```

with direction/magnitude decompositions retained only as diagnostics/loss terms.

If full-vector compatibility also fails, then the next required evidence is richer than a single carrier motion vector—most likely local-neighborhood deformation/Jacobian or a candidate-conditioned cross-view relation model.

## Authorization boundary

- F16 H: remains frozen and supported.
- p_active typed evidence: remains supported.
- standalone/global/H-relative amplitude: falsified.
- candidate-specific amplitude-only local motion energy: falsified.
- full-vector deterministic candidate compatibility: next authorized diagnostic.
- no head/backbone training.
- no large/end-to-end training.

## Reproducibility

Exact aggregate local result SHA-256:
`feedbc11f74b78b39f7d092bf130596a5f430b6934ec79df482e00fdb4163efc`

Exact source SHA-256:
`8a51e3717ff26a1b3c5d8c2ec65b4b75cdd970dfdd35bb6d50cb6f03f33febe2`

Canonical source: `candidate_specific_local_motion_energy_v1.py`.
