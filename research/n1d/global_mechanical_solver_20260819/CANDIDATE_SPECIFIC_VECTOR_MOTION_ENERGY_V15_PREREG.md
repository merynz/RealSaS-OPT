# RealSaS N1D — Candidate-Specific Full-Vector Motion Energy V1.5 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DIAGNOSTIC__NO_TRAINING`  
**Parent amplitude-only result:** `FAIL__AMPLITUDE_ONLY_LOCAL_MOTION_COMPATIBILITY_NOT_FAMILY_ROBUST`  
**Frozen geometry authority:** F16 Bounded H Research Contract V1

## Causal question

Amplitude-only candidate energy failed even though several fields retained nontrivial candidate-pair ordering. Under orthographic projection, 2D displacement magnitude depends jointly on the 3D displacement magnitude and direction.

V1.5 therefore asks:

> Does the same frozen local motion evidence become family-robust when each feasible H candidate is scored by its **full projected 2D motion vector**, rather than forcing amplitude and direction to be independent candidate energies?

No new observation channel is introduced.

## Panel

Same 264 reliable truth-active F16-contained carriers from P0.5/V1, over:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Truth is evaluation-only. H/model/fields remain frozen.

## Candidate motion

For candidate `h`, carrier `i`, usable Pose-A view `v`:

```text
d_h(i,v) = project(h,v) - project(P_A(i),v)
```

Observation vectors are exactly the V1 frozen fields:

- native raster DIS A->B flow;
- learned `transport_offset_srcA` converted by `255/31`;
- per-view `delta_point_map_srcA` projected into the view.

## Energy family 1 — full vector L2

For each field:

```text
E_vec(i,h) = median_v ||d_h(i,v) - d_obs(i,v)||_2
```

Require >=2 usable views. Lower is better.

Arms:

```text
V_DIS
V_TRANSPORT
V_DELTA3D
```

## Energy family 2 — joint polar compatibility

For a view:

```text
r_amp = |log(eps + ||d_h||) - log(eps + ||d_obs||)|
```

If both magnitudes are >=1 native pixel:

```text
r_dir = 1 - cosine(d_h,d_obs)
```

otherwise `r_dir=0` so very small motion is not assigned arbitrary angle.

Joint energy:

```text
E_polar(i,h) = median_v (r_amp + r_dir)
```

Equal unit weight is frozen; no fitted mixing coefficient.

Arms:

```text
P_DIS
P_TRANSPORT
P_DELTA3D
```

## Deterministic multi-evidence fusion

For the three `V_*` arms separately, robust-normalize candidate energies within each carrier H by median/IQR and take candidate-wise median:

```text
V_FUSION
```

Do the same for `P_*`:

```text
P_FUSION
```

No learned/family-specific weights.

## Endpoint evaluation

For each arm choose:

```text
h_sel = argmin_h E(i,h)
```

Let `h*` be evaluator-only H candidate nearest truth P_B. Report selected endpoint quality against truth P_B:

```text
contain_1x = ||h_sel - truth_P_B|| <= local_B_scale
contain_2x = ||h_sel - truth_P_B|| <= 2*local_B_scale
normalized_endpoint_error = ||h_sel-truth_P_B|| / local_B_scale
```

Report pooled and per-family:

- contain_1x
- contain_2x
- median normalized endpoint error
- p90 normalized endpoint error

Also report selected candidate amplitude within25/within50 relative to the oracle-near H candidate amplitude as a secondary metric.

## Promotion gates

A deterministic full-vector candidate energy supports the joint candidate-motion compatibility abstraction only if all hold:

```text
pooled contain_2x >= 0.75
worst-family contain_2x >= 0.60
13203 contain_2x >= 0.60
15290 contain_2x >= 0.60
pooled contain_1x >= 0.55
worst-family contain_1x >= 0.40
pooled median normalized endpoint error <= 0.90
```

Tail gap guard:

```text
best-family contain_2x - worst-family contain_2x <= 0.25
```

These gates are intentionally below the H-oracle ceiling; V1.5 tests whether a deterministic local motion relation is strong enough to become an evidence term, not whether it replaces the compiler/global solver.

## Decision

1. If a single-field vector/polar arm passes, prefer the simplest passing field unless its corresponding fusion improves worst-family contain_2x by >=.05 with no new gate failure.
2. If only fusion passes, support deterministic joint local-motion compatibility as a multi-evidence term.
3. If full-vector passes while amplitude-only V1 failed, replace the independent candidate-scoring concept `log_amp + dir` with:

```text
p_active + candidate-conditioned vector/differential compatibility
```

Amplitude and direction may remain separate supervision/diagnostics, but not independent final candidate authorities.
4. If no V1.5 arm passes, a single-carrier motion vector is insufficient. Next representation work must use a richer **local relational/deformation field**: neighborhood displacement/Jacobian, candidate-conditioned cross-view relation tokens, or equivalent differential mechanics.

## Boundaries

- no new model weights;
- no descriptor Z fusion;
- no truth-conditioned energy;
- no free XYZ;
- no head/backbone training;
- no qualification claim;
- large/end-to-end training remains forbidden.
