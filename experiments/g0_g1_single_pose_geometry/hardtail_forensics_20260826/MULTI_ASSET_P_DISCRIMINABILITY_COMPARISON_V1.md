# RealSaS P Hard-Tail — Multi-Asset CPU Discriminability Comparison V1

**Date:** 2026-08-26  
**Status:** `EXPLORATORY_COMPARATIVE__NOT_FORMAL_GATE`  
**Execution:** assistant-side CPU/container; no Colab; no optimizer/training steps.  
**Splits:** only already-open FIT/proxy assets; no TUNE/CAL/DEV/EXTERNAL opening.

## Question

Is the gauge-safe P hard tail one common low-texture/planar failure, or do different assets fail for different observable reasons? Does a known-camera RGB/alpha hypothesis test already contain enough evidence to beat the trained direct P predictor?

## Assets

Three gauge-safe hard-tail witnesses were compared with the same deterministic candidate-depth assay:

| key | asset | frozen model P95@512 | visual / geometric note |
|---|---|---:|---|
| f089 | `asset_f089abadcd071194617d640b` | 0.448743 | plant-like asset plus detached giant 2-triangle rectangular component |
| ea593 | `asset_ea593d044e14f20abe6d2818` | 0.343100 | long low-poly creature/object; severe axial foreshortening at end views |
| 662ed | `asset_662ed7f1e328bd85959157cf` | 0.249118 | articulated deer/antler geometry; ordinary riggable-character-like witness |

Geometry audit confirms f089 is structurally exceptional: its largest triangle is ~850× its median triangle area; ea593 and 662ed are ~7.9× and ~9.9× respectively.

## Common inference-side evidence

For every reference query:

```text
reference raster XY
+ known orthographic camera
+ candidate forward depth d in 321 samples over [-0.54,+0.54]
        -> candidate world P(d)
        -> reproject into other views
        -> RGB + alpha consistency cost
```

Teacher geometry/barycentrics are used **only after scoring** to evaluate the true forward depth. Teacher visibility is absent from the main inference assay.

Cardinal references V0/V2/V4/V6 were used for matched sampling (1400 loci/view, 5600 loci/asset). All eight render images were available as target evidence.

## Result 1 — adding all views helps f089 dramatically, but does not generalize

| asset | direct model P95 | all-view mean7 P90 | mean7 P95 | best raw-RGB P95 mode | best raw-RGB P95 |
|---|---:|---:|---:|---|---:|
| f089 | 0.448743 | **0.100459** | 0.347967 | best4 | **0.262374** |
| ea593 | **0.343100** | 0.394465 | 0.551166 | perp2_mean | 0.483285 |
| 662ed | **0.249118** | 0.230231 | 0.310406 | best6 | 0.283221 |

The f089 result is consistent with the previous plane/propagation diagnosis: additional baselines expose useful geometry and sharply reduce most of the distribution. But the same raw photometric geometry loop is **worse than the trained direct predictor at P95** on ea593 and 662ed.

This falsifies the simple treatment:

```text
KNOWN CAMERAS + RAW RGB REPROJECTION == GENERAL HARD-TAIL FIX
```

## Result 2 — aggressive view selection is not a universal remedy

All-view aggregations:

| asset | mean7 P95 | best6 P95 | best4 P95 | median7 P95 | median7 truth top8 |
|---|---:|---:|---:|---:|---:|
| f089 | 0.347967 | 0.344731 | **0.262374** | 0.264408 | 0.8496 |
| ea593 | 0.551166 | 0.549237 | **0.500220** | 0.503268 | 0.6452 |
| 662ed | 0.310406 | **0.283221** | 0.320105 | 0.320105 | **0.7798** |

The optimal aggregation differs by asset. This repeats the f089 lesson that view-selection policy itself can destroy useful constraints; no fixed `best-N` is justified yet.

## Result 3 — silhouette geometry contains truth, but usually not a singleton depth

An image-only alpha/visual-hull assay asked which candidate depths remain inside foreground support across the other views.

| asset | truth inside silhouette-feasible set | median feasible depth count / 321 | feasible width P50 | feasible width P90 |
|---|---:|---:|---:|---:|
| f089 | 0.4304 | 4 | 0.08775 | 0.24638 |
| ea593 | **0.9380** | **93** | **0.31050** | **0.96525** |
| 662ed | **0.9452** | **63** | **0.31050** | **0.48600** |

For ea593 and 662ed, the true depth is almost always geometrically legal under the silhouettes, but dozens of depths remain legal. The visual hull is therefore a useful **retention/corridor constraint**, not a sufficient depth selector.

The low-texture hypothesis also does not generalize. In the all-view best6 slice, 662ed low-texture P90 is ~0.106 versus ~0.195 on higher-texture loci; ea593 is roughly comparable across slices. f089 remains the special case where the giant uniform plane is genuinely problematic.

## Result 4 — oracle visibility selection does not close the representative tails

A separate diagnostic used teacher visibility **only as an oracle** over the cardinal target views. This improves truth top-k rank in places but does not materially collapse P95:

| asset | cardinal mean P95 | teacher-visible oracle P95 | oracle top8 |
|---|---:|---:|---:|
| f089 | 0.398845 | 0.815554 | 0.8405 |
| ea593 | 0.482245 | 0.534659 | 0.6360 |
| 662ed | 0.341466 | 0.341064 | 0.6967 |

The f089 oracle P95 is pathological because many queries retain only one appearance-ambiguous visible target; the ranking improvement without argmin improvement is itself evidence that raw appearance has broad repeated minima. More importantly, ea593/662ed do not become solved when occluded views are removed. **Occlusion/view selection is not the sole missing mechanism.**

## Cross-asset diagnosis

### f089 — `LOW_TEXTURE_PLANAR_PROPAGATION + SUPPORT_OWNERSHIP`

The previous teacher-free region experiment remains positive: once the dominant plane region is discovered and confident image-derived seeds are propagated geometrically, the large plane tail collapses to ~1e-3 P90. The new common assay confirms that additional baselines matter. f089 is a valid positive control for geometry propagation, but it is not representative of every hard tail because of its detached giant quad.

### ea593 — `AXIAL_FORESHORTENING + CORRESPONDENCE / FEATURE AMBIGUITY`

The long object is nearly end-on in V0/V4. Silhouettes retain the true depth but leave a huge interval (median 93 candidate depths; P90 width ~0.965). Raw RGB hypothesis testing is worse than the trained model. Low texture alone does not explain the error, and visibility-oracle selection does not solve it.

Interpretation: the missing operation is not merely `test more candidate depths`; it is **identify which cross-view surface evidence belongs to the query under strong viewpoint change**.

### 662ed — `MULTISURFACE / VIEW-DEPENDENT CORRESPONDENCE HARD TAIL`

This ordinary character-like deer is the most important counterexample to the f089-only story. Raw all-view evidence improves median/P90 under some view aggregation, and median aggregation retains truth in top8 ~78%, but P95 remains worse than the current trained model. Low-texture loci are not the hardest slice.

Interpretation: useful cross-view evidence exists, but raw color equality is not the right discriminator for thin antlers, overlapping limbs/surfaces, cel-shading changes and repeated same-colored regions.

## Main decision

The hard tail is **heterogeneous**.

What survives:

```text
known camera + candidate P + cross-view verification
```

What is falsified as a general solution:

```text
raw RGB/alpha photometric cost + fixed view aggregation
```

The especially important observation is that on ea593 and 662ed the trained direct predictor already beats the raw RGB geometry oracle at P95. Therefore a geometry-in-the-loop treatment must consume **learned / appearance-invariant IRIS features (and possibly N/U/support)** rather than replacing learned evidence with raw photometric matching.

## Next experimental rung

`D2 = LEARNED-FEATURE REPROJECTION ORACLE` on ea593 + 662ed, with f089 retained as a positive-control morphology.

Required comparison:

```text
D1 raw RGB/alpha reprojection          [measured here]
D2 frozen learned-feature reprojection [NEXT]
D3 + normal/plane patch hypothesis     [only if D2 leaves structured tail]
D4 visibility-aware source selection   [only if causal after D2]
D5 multi-scale geometric consistency   [reserve]
```

Do **not** authorize full PatchMatch or longer P training from this experiment alone.
