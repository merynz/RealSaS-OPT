# RealSaS N1D — Candidate-Specific Local Motion Energy V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DIAGNOSTIC__NO_BACKBONE_TRAINING`  
**Parent result:** `CANDIDATE_CONDITIONED_AMPLITUDE_P05_RESULT.json`  
**Frozen geometry authority:** F16 Bounded H Research Contract V1

## Question

P0 and P0.5 falsified a standalone carrier-level amplitude scalar under the tested evidence contracts. V1 asks whether the missing information is present **before scalar consensus**, as local per-view A->B motion evidence that can score each feasible H candidate directly.

For carrier `i`, candidate `h in H_i`, usable view `v`:

```text
d_h(i,v) = project(h,v) - project(P_A(i),v)   # native 256px coordinates
```

The diagnostic compares `d_h` with observation-native local motion at the Pose-A projection. Truth is never used in an energy definition.

Evaluation is restricted to the same reliable, truth-active, F16-contained carriers used by P0.5. Truth is used only after energies are frozen to identify the evaluator-only oracle-near H candidate and score magnitude compatibility.

## Frozen observation fields

### DIS raster flow

OpenCV DIS A->B flow computed directly from the paired native 256px rasters, using the same historical Problem-A implementation. Bilinearly sample the 2D flow at `project(P_A,v)`.

Units: native pixels, x-right/y-down.

### Learned transport offset

Sample frozen `transport_offset_srcA` at the Pose-A projection. The field is defined on the N1D f4 lattice and is in f4 grid-cell units. Convert to native-pixel displacement exactly by:

```text
d_transport_px.x = offset_x * (255 / (W_f4 - 1))
d_transport_px.y = offset_y * (255 / (H_f4 - 1))
```

For the frozen model `H_f4=W_f4=32`.

### Per-view differential 3D field

Sample frozen `delta_point_map_srcA(v)` at Pose-A. Convert it to the corresponding view displacement by projecting both `P_A` and `P_A + delta_v` into the same camera:

```text
d_delta3d_px(v) = project(P_A + delta_srcA(v),v) - project(P_A,v)
```

This preserves per-view differential evidence; no cross-view scalar consensus is taken.

## Magnitude-only candidate energies

Direction is deliberately excluded from the primary amplitude test.

For any observed view-motion vector `d_obs` and candidate vector `d_h`, define:

```text
r_amp = | log(eps + ||d_h||) - log(eps + ||d_obs||) |
```

Aggregate over Pose-A usable views with the **median**. Views with non-finite evidence are ignored; require >=2 views.

Lower energy is better.

Arms:

```text
A_DIS        median r_amp using raw DIS
A_TRANSPORT  median r_amp using learned transport_offset_srcA
A_DELTA3D    median r_amp using per-view delta_point_map_srcA projection
```

### A_ROBUST_FUSION

No fitted weights. For each carrier, robust-normalize each available arm's candidate energies by its within-H median and IQR:

```text
z_E = (E - median_H(E)) / max(IQR_H(E), eps)
E_fused = median(z_E_DIS, z_E_TRANSPORT, z_E_DELTA3D)
```

This is deterministic and family-independent.

## Direction-only diagnostics

For DIS / transport / delta3D separately, report but do not use for amplitude promotion:

```text
E_dir = median_v(1 - cosine(d_h, d_obs))
```

Ignore views where either magnitude is below 1 native pixel. Direction diagnostics exist only to determine whether a later candidate energy should keep amplitude/direction factored.

## Evaluation metrics — amplitude only

Let `h*` be the evaluator-only H candidate nearest truth P_B and `a*=||h*-P_A||`.

For each amplitude energy:

1. choose `h_amp = argmin_h E_amp(i,h)`;
2. compare `||h_amp-P_A||` with `a*`;
3. report per family and pooled:

```text
within25 = fraction max(a_sel,a*)/min(a_sel,a*) <= 1.25
within50 = fraction ratio <= 1.50
median_abs_log_ratio = median |log((a_sel+eps)/(a*+eps))|
```

Candidate-pair ordering metric:

For pairs of H candidates whose absolute log-amplitude error to `a*` differs by at least `log(1.10)`, report the fraction for which lower energy is assigned to the candidate with lower oracle amplitude error. To bound combinatorics, deterministically evaluate at most 4096 uniformly index-spaced candidate pairs per carrier.

Also report the percentile rank of the oracle-nearest candidate amplitude under energy after collapsing H candidates into log-amplitude bins of width `log(1.05)`; this avoids rewarding duplicate triangulation hypotheses.

## Gates for amplitude-channel support

An arm supports a candidate-specific amplitude channel only if all hold:

```text
pooled within25 >= 0.70
worst-family within25 >= 0.55
pooled within50 >= 0.85
worst-family within50 >= 0.75
mean family pairwise ordering >= 0.70
worst-family pairwise ordering >= 0.60
pooled median abs log-ratio <= log(1.25)
```

Hard-tail guard:

```text
13203 within25 >= 0.55
15290 within25 >= 0.55
```

No pooled success may hide either historical amplitude tail.

## Decision

1. If a single observation field passes, prefer the simplest passing field unless robust fusion improves worst-family within25 by >=.05 without degrading pairwise tail.
2. If only robust fusion passes, support deterministic multi-evidence candidate amplitude compatibility.
3. If no amplitude arm passes, current paired-raster local motion front door is not sufficient under these observation fields; do not train amplitude. Next work must add/learn a richer local differential relation (e.g. neighborhood deformation/Jacobian or candidate-conditioned cross-view relation features).

## Boundaries

- `p_active` remains separate and supported from P0.
- amplitude and direction remain causally separated in V1.
- no descriptor Z fusion;
- no truth-conditioned energy;
- no free XYZ;
- no backbone/head training;
- no qualification claim;
- large/end-to-end training remains forbidden.
