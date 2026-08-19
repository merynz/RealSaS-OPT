# Representation Sufficiency Battery V1 — Test-C Mapping Addendum

**Date:** 2026-08-19  
**Status:** `PREREG_ADDENDUM_BEFORE_CURRENT_H_REPLAY`  
**Parent prereg commit:** `b87e222fdf7bf4c66a62314c88ee9e155a576c73`

## Why an addendum is required

The current descriptor-global-foreground route constructs raster-derived Problem-A carriers `P_A`; these carriers do not carry hidden teacher surface IDs. Test C therefore needs a frozen evaluator-only mapping from each reconstructed carrier to the paired observable surface target before current feasible hypothesis set `H_i` is inspected.

This addendum is committed **before** the current-route `H_i` replay results are computed.

## Evaluator-only carrier mapping

For each episode sidecar, transform observable target surface points into the same normalized camera/world coordinates used by the raster front door:

```text
P*_A_norm = (surface_points_A - camera_center) / (2 * camera_half_extent)
P*_B_norm = (surface_points_B - camera_center) / (2 * camera_half_extent)
```

For each raster-derived Problem-A carrier `P_A[i]`, choose the nearest `P*_A_norm[j]`. This target index is evaluator-only and is never used by the candidate generator or solver.

## Local surface scale

For every target surface sample `j`, define local scale as the median Euclidean distance to its four nearest *other* surface samples in the same pose:

```text
s_A[j] = median(4 nearest-neighbor distances around P*_A_norm[j])
s_B[j] = median(4 nearest-neighbor distances around P*_B_norm[j])
```

## Mapping reliability

A carrier is included in the primary containment denominator only when:

```text
||P_A[i] - P*_A_norm[j]|| <= 2 * s_A[j]
```

The fraction excluded by this mapping-reliability guard must be reported. A family with poor mapping reliability cannot be declared GREEN from this test.

## Current feasible basin

Instrument the **unchanged current descriptor-global-foreground solver** to export all valid pairwise rank-3 hypotheses `H_i` generated before mean-reprojection selection and all-view nearest-candidate LS refit. Exporting `H_i` must not alter candidate search, scoring, committed endpoint, fallback, or model outputs.

## Containment metrics

For mapped target endpoint `P*_B_norm[j]`, define:

```text
e_i = min_{h in H_i} ||h - P*_B_norm[j]||
```

Report both:

```text
strict containment:  e_i <= 1 * s_B[j]
primary containment: e_i <= 2 * s_B[j]
```

If `H_i` is empty, containment is false. Also report continuous normalized error `e_i / s_B[j]`, hypothesis count, solver success, and committed-endpoint normalized error.

## Tail decision

The parent prereg Test-C RED rule applies to **primary (`2×`) containment**:

- hard/worst family or difficulty stratum `<0.90` containment, when denominator is meaningful; or
- >=15 percentage-point containment loss relative to easy/best stratum.

The `1×` result is a stricter diagnostic and is not substituted post hoc for the frozen `2×` primary decision.

Current-route results from this addendum remain truth-open development evidence and cannot modify the frozen Stage-B qualification.
