# RealSaS-OPT — Current State

**Date:** 2026-08-26  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `P_SCALE_SIGNAL_NOT_STRONG__GAUGE_PATHOLOGY_LOCALIZED__MULTI_ASSET_DISCRIMINABILITY_COMPLETE__LEARNED_FEATURE_REPROJECTION_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question is the **single-pose observable common-frame P hard tail**, specifically camera-forward depth under known cameras.

## Current P formulation

P-V5/R256 uses analytic screen-plane coordinates from raster XY + known camera and learns only camera-forward depth:

```text
raster x/y + known camera -> analytic screen-plane P
image evidence            -> learned forward depth
                                  |
                                  v
                              full P
```

## Frozen FIT-scale ladder

Same architecture, no augmentation change, 7168 successful optimizer updates per rung:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

Verdict: `SCALE_SIGNAL_BUT_NOT_STRONG`.

512/32 aggregate ratio = `0.68800194`; median ratio = `0.38632081`.

## Gauge localization V3 — COMPLETE

Decision: `TARGET_GAUGE_PATHOLOGY_ESTABLISHED__CORRECT_BEFORE_OPTIMIZATION_OR_PATCHMATCH`

- PROXY32 analytically impossible: `2/64`
- PROXY32 gauge-safe: `62/64`
- TRAIN512 analytically impossible: `6/1024`
- repeated micro-P95 `28.4803` spikes share `asset_90b5f2cc531696040da996a9`

Interpretation: severe target/gauge pathologies exist but explain only a small minority of the proxy hard tail.

## f089 positive-control forensic

`asset_f089abadcd071194617d640b` is gauge-safe but contains a detached giant rectangle (4 vertices / 2 faces / zero source skin weight on those vertices). Its semantic role remains unresolved.

Teacher-free image-region discovery + cross-view candidate scoring + robust planar propagation is an exploratory positive on the giant low-texture plane: propagated P90 reaches roughly `0.0011–0.0015` on non-edge-on views. The failed first `best-3` view-selection attempt is preserved in the archive; using all seven target views restored the missing geometric constraints.

Interpretation: f089 proves a real `LOW_TEXTURE_PLANAR_PROPAGATION + SUPPORT_OWNERSHIP` mechanism, but must not be treated as the only hard-tail morphology.

## Multi-asset P discriminability comparison V1 — COMPLETE

Canonical report:
`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/MULTI_ASSET_P_DISCRIMINABILITY_COMPARISON_V1.md`

Three gauge-safe hard-tail witnesses were tested with a common CPU candidate-depth sweep (321 depths, 5600 loci/asset, known cameras, image-only RGB/alpha inference evidence; teacher geometry only for scoring):

| asset | frozen direct model P95@512 | all-view mean7 P95 | best raw-RGB P95 |
|---|---:|---:|---:|
| f089 | 0.448743 | 0.347967 | **0.262374** (`best4`) |
| ea593 | **0.343100** | 0.551166 | 0.483285 (`perp2_mean`) |
| 662ed | **0.249118** | 0.310406 | 0.283221 (`best6`) |

Key result: raw known-camera photometric hypothesis testing improves f089 but is **worse than the trained direct model at P95 on ea593 and 662ed**.

Silhouette/visual-hull oracle:

- ea593 truth feasible `0.9380`, median feasible depth count `93/321`, feasible width P90 `0.96525`
- 662ed truth feasible `0.9452`, median feasible depth count `63/321`, feasible width P90 `0.48600`

Thus silhouettes strongly retain truth but usually do not identify a singleton depth.

Teacher-visibility-only selection does not close ea593/662ed, so occlusion/view selection is not the sole missing mechanism.

### Current hard-tail taxonomy

- `f089`: low-texture planar propagation + region/support ownership; special giant-quad morphology.
- `ea593`: axial foreshortening + cross-view correspondence / appearance-feature ambiguity.
- `662ed`: multisurface / view-dependent correspondence hard tail (thin antlers, overlap, repeated colors/cel shading); low texture is not the dominant slice.

**Conclusion:** hard tail is heterogeneous. `known camera + candidate P + cross-view verification` survives, but `raw RGB/alpha + fixed view aggregation` is falsified as the general treatment.

The direct predictor already beats raw RGB geometry verification on the representative ea593/662ed tails, so the next geometry-in-the-loop oracle must consume **frozen learned / appearance-invariant IRIS features**, not replace them with photometric matching.

## Canonical experiment archive

`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/`

Important chronology:

1. `P_HARDTAIL_FORENSICS_20260826.md`
2. `F089_CPU_DISCRIMINABILITY_ORACLE_PRELIMINARY.md`
3. `F089_TEACHER_FREE_REGION_PROPAGATION_V1.md`
4. `MULTI_ASSET_P_DISCRIMINABILITY_COMPARISON_V1.md`

The archive deliberately preserves failed intermediate hypotheses and causal corrections.

## NEXT EXECUTABLE STEP

`D2 = LEARNED-FEATURE REPROJECTION ORACLE`

Primary witnesses:
- `asset_ea593d044e14f20abe6d2818`
- `asset_662ed7f1e328bd85959157cf`

Positive-control morphology:
- `asset_f089abadcd071194617d640b`

Required ladder:

```text
D1 raw RGB/alpha reprojection          [COMPLETE]
D2 frozen learned-feature reprojection [NEXT]
D3 + normal/plane patch hypothesis     [conditional]
D4 visibility-aware source selection   [conditional]
D5 multi-scale geometric consistency   [reserve]
```

The D2 experiment must use frozen current IRIS features/checkpoint and known cameras without training or retuning on these witness outcomes. It must report truth rank/top-k and depth P50/P90/P95 against the D1 baselines above.

## Authorization state

`GAUGE_LOCALIZATION = COMPLETE`

`F089_TEACHER_FREE_ORACLE = EXPLORATORY_POSITIVE`

`MULTI_ASSET_D1_RGB_ORACLE = COMPLETE`

`D2_LEARNED_FEATURE_REPROJECTION = NEXT`

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
