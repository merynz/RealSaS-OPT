# RealSaS-OPT — Current State

**Date:** 2026-08-26  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `P_SCALE_SIGNAL_NOT_STRONG__GAUGE_PATHOLOGY_LOCALIZED__F089_TEACHER_FREE_PROPAGATION_POSITIVE__REPRESENTATIVE_ORACLE_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question is no longer the historical M256 matcher gate. Current work has moved to the **single-pose observable common-frame P hard tail**, specifically camera-forward depth under known cameras.

## Current P formulation

P-V5/R256 does not freely regress XYZ. It uses analytic screen-plane coordinates from the raster location and known camera, and learns camera-forward depth.

Approximate decomposition:

```text
raster x/y + known camera -> analytic screen-plane P
image evidence            -> learned forward depth
                                  |
                                  v
                              full P
```

Therefore full-P error must be localized into analytic screen/gauge error versus learned forward-depth/discriminability error.

## Frozen FIT-scale ladder

Same architecture, no augmentation change, 7168 successful optimizer updates per rung:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

Verdict: `SCALE_SIGNAL_BUT_NOT_STRONG`.

512/32 aggregate ratio = `0.68800194`; median ratio = `0.38632081`.

The 512 rung is partly optimization-budget limited, but the stationary worst tail required separate localization.

## Gauge localization V3 — COMPLETE

Decision:

`TARGET_GAUGE_PATHOLOGY_ESTABLISHED__CORRECT_BEFORE_OPTIMIZATION_OR_PATCHMATCH`

Counts:

- PROXY32 analytically impossible: `2/64`
- PROXY32 gauge-safe: `62/64`
- TRAIN512 analytically impossible: `6/1024`

Repeated micro-P95 `28.4803` spikes at steps `2752`, `4768`, `6240` share:
`asset_90b5f2cc531696040da996a9`.

Interpretation: a small number of severe target/gauge pathologies contaminated the apparent hard tail, but most proxy cells remain genuine gauge-safe residuals.

No DEV/TUNE/CAL/EXTERNAL split was opened; scientific optimizer steps during the gauge audit = 0.

## f089 gauge-safe witness

`asset_f089abadcd071194617d640b`:

- analytic screen P95 ≈ `7.3e-05`
- full-P P95 at rung512 ≈ `0.408874` cel / `0.448743` ink

Thus its ~0.4 residual is not screen-gauge error.

### Important asset caveat

The asset contains a real detached rectangular component:

- 4 vertices: `591..594`
- 2 faces: `776,777`
- disconnected from the rest of the mesh
- all four vertices have source skin-weight sum `0.0`
- face area ≈ `0.148846` each

This explains the giant brown rectangle in several views. Its semantic role is unresolved; do not silently label it junk, but do not treat f089 as the sole representative of normal product hard tail.

## f089 CPU discriminability oracle

### Preliminary, teacher-region-assisted

On the huge low-texture plane, pointwise depth was ambiguous. Image-derived confident seeds plus teacher-provided plane membership and robust planar propagation collapsed catastrophic tails (e.g. V2 ~`0.3606 -> 0.00274`).

### Teacher-free follow-up

Teacher triangle membership and teacher visibility were removed from inference.

Image-only dominant-region discovery reached high IoU on non-edge-on views (V1/V2/V3/V5/V6/V7 ≈ `0.889–0.992`).

The first teacher-free `best-3` view-selection attempt **FAILED** because it discarded edge-on V0/V4 constraints.

Using all seven target views restored the informative geometry. Seed correctness was `95–100%`, and propagated P90 became:

- V1 `0.001113`
- V2 `0.001253` from pointwise `0.359339`
- V3 `0.001460`
- V5 `0.001059`
- V6 `0.001526` from pointwise `0.046122`
- V7 `0.001345`

This is an exploratory mechanism result, not a formal preregistered PASS.

Remaining f089 error is concentrated in image-region contamination / surface-support ownership, not missing plane-depth information.

## Canonical experiment archive

`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/`

Read in order:

1. `P_HARDTAIL_FORENSICS_20260826.md`
2. `F089_CPU_DISCRIMINABILITY_ORACLE_PRELIMINARY.md`
3. `F089_TEACHER_FREE_REGION_PROPAGATION_V1.md`
4. `F089_TEACHER_FREE_REGION_PROPAGATION_V1_METRICS.json`

The archive intentionally preserves the apparent catastrophe, the failed first teacher-free view-selection attempt, and the causal correction.

## NEXT EXECUTABLE STEP

Run the same P discriminability ladder on a **gauge-safe hard-tail asset without the detached giant-quad pathology**.

Frozen proxy candidate:
`asset_ea593d044e14f20abe6d2818`

Questions:

1. Does truth depth remain discriminable under cross-view reprojection?
2. Is the failure low-texture/planar propagation, visibility/view selection, feature evidence, or direct regression?
3. Does the diagnosis survive on ordinary riggable/object geometry?

Only after this representative witness may a minimal geometry-in-the-loop treatment be preregistered.

## Authorization state

`GAUGE_LOCALIZATION = COMPLETE`

`F089_TEACHER_FREE_ORACLE = EXPLORATORY_POSITIVE`

`REPRESENTATIVE_GAUGE_SAFE_ORACLE = NEXT`

`PATCHMATCH = NOT_AUTHORIZED`

`LONGER_P_TRAINING = NOT_NEXT`

`CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

## Research rule

```text
apparatus/target gauge
 -> observation discriminability
 -> representation
 -> learner/optimizer
 -> evidence consumer / geometric verification
 -> only then information limit
```
