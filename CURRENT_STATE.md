# RealSaS-OPT — Current State

**Date:** 2026-08-27  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_OBSERVABLE_GEOMETRY_PREREGISTERED__D2_D5_CORRESPONDENCE_BOTTLENECK_LOCALIZED__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question is the **single-pose observable common-frame P hard tail**, specifically whether the actually observable eight-view surface geometry is sufficient and whether deterministic persistence can preserve it before any new raster representation is blamed.

## Frozen prior evidence

P-V5/R256 uses analytic screen-plane coordinates from raster XY + known camera and learns only camera-forward depth.

Frozen FIT scale ladder, same architecture/objective:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

The median improves strongly with scale while the worst-cell tail is effectively stationary. This alone is not a proof of missing representation prior, but it makes “just add more of the same scratch training data” a weak next hypothesis.

Frozen rung512 checkpoint SHA-256:
`a92fa18c46975578f2705ee3baa426cce573d4a767d42490e7599ed905f6e48e`.

Gauge localization V3 established a small number of target/gauge pathologies but did not explain the representative hard tail.

## D1 / D2→D5 localization — COMPLETE

Canonical report:
`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/P_HARDTAIL_ORACLE_D2_D5_20260826.md`

Representative gauge-safe witnesses remain:

- `f089 = asset_f089abadcd071194617d640b`: low-texture giant-plane positive control;
- `ea593 = asset_ea593d044e14f20abe6d2818`: severe foreshortening/correspondence hard tail;
- `662ed = asset_662ed7f1e328bd85959157cf`: thin/multisurface/repeated-appearance hard tail.

D2 frozen learned-feature reprojection:

| asset | matched direct P95 | best D2 P95 |
|---|---:|---:|
| f089 | 0.416423 | **0.123106** |
| ea593 | **0.471771** | 0.574262 |
| 662ed | **0.271590** | 0.304407 |

D3 exact-normal plane warp did not close ea593/662ed. D4 teacher visibility/source-view selection did not close them. D5 coarse-to-fine feature consistency pruned truth too early:

- ea593: s8 top64 `0.618` -> after s4 top16 `0.253`;
- 662ed: s8 top64 `0.840` -> after s4 top16 `0.338`.

D2b tiny 3-asset learned pair metric was negative on ea593/662ed and is too small to prove the representation itself lacks information.

**Strongest current localization:** representative gauge-safe failures are bottlenecked **before or at cross-view correspondence evidence/ranking**. Search cannot recover truth that the evidence has already ranked away. This is not an information-theoretic impossibility claim and not yet proof that a pretrained encoder is required.

## E0 — CURRENT EXECUTABLE GATE

Preregistration and apparatus:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/`

E0 removes the raster learner entirely. Both arms receive the **same exact view-local visible common-frame P** from native 1024 raster authority. The only treatment variable is persistence/correspondence authority.

```text
E0-a  oracle persistence
      teacher physical surface identity / visibility witness
      -> upper bound under perfect persistence

E0-b  deterministic SurfaceBuilder persistence
      only exact common-frame P
           + P-derived local normal continuity
           + known orthographic view geometry
           + reciprocal reprojection/cycle
           + raster pixel provenance
      -> what SurfaceBuilder can do without teacher identity
```

E0-b teacher triangle/bary identity is evaluation-only and physically absent from the matching API. `camera.json` is not consumed. No Pose B, joints, parents, skin weights, owner IDs, mechanics/GFDR or compiler IDs enter construction.

Population is inherited prospectively from the already-frozen FIT scale ladder:

- 512 FIT train-order authority for later fixed downstream probes;
- historical disjoint 8-asset FIT calibration panel;
- frozen family-disjoint `FIT_PROXY32` qualification panel;
- TUNE/CAL/DEV/EXTERNAL closed.

The geometry-stage apparatus has local compile/unit/firewall closure; no E0 scientific outcome has been opened yet.

## E0 decision tree

```text
E0-a observable-only geometry downstream sufficient?
  NO  -> visible-only substrate is insufficient under the tested downstream contract;
         do not jump to pretrained correspondence priors.
  YES -> evaluate E0-b.

E0-b non-inferior to E0-a?
  NO  -> SurfaceBuilder persistence is a blocker even with exact P;
         localize persistence before blaming the encoder.
  YES -> exact-P persistence is operationally sufficient;
         advance to the frozen three-arm raster correspondence intervention.
```

Downstream non-inferiority margins must be frozen on the calibration 8 before `FIT_PROXY32` is opened for E0 qualification outcome inspection.

## AFTER E0 ONLY — frozen three-arm correspondence intervention

Exactly three matched arms:

```text
C1  scratch + current objective
    -> current baseline

C2  scratch + tail-aware correspondence objective
    -> cheap control for objective/curriculum mismatch

C3  pretrained prior transplant
    -> RealSaS-native decoder with pretrained visual/multiview prior
       (MapAnything-family prior is a candidate)
       NOT the full MapAnything camera/ray/product wrapper
```

The scientific question is whether the hard-tail correspondence evidence/ranking improves, not whether a larger model can fit the training set.

Only if a representation demonstrates correct correspondence ranking/top-k containment on frozen representative tails may multiview search / fusion / propagation / PatchMatch-style mechanisms be promoted.

## Authorization state

`GAUGE_LOCALIZATION = COMPLETE`

`MULTI_ASSET_D1_RGB_ORACLE = COMPLETE`

`D2_LEARNED_FEATURE_REPROJECTION = COMPLETE_NEGATIVE_REPRESENTATIVE__POSITIVE_F089`

`D3_EXACT_NORMAL_PLANE = COMPLETE_NO_CLOSURE`

`D4_VISIBILITY_ORACLE = COMPLETE_NO_CLOSURE`

`D5_MULTISCALE = COMPLETE_NO_CLOSURE`

`D2B_3ASSET_METRIC = EXPLORATORY_NEGATIVE_REPRESENTATIVE`

`E0_OBSERVABLE_GEOMETRY = AUTHORIZED_CURRENT`

`E0_SCIENTIFIC_OUTCOME = NOT_YET_OPENED`

`C1_SCRATCH_CURRENT = NOT_YET_EXECUTED`

`C2_SCRATCH_TAIL_AWARE = NOT_YET_EXECUTED`

`C3_PRETRAINED_PRIOR_TRANSPLANT = NOT_YET_EXECUTED`

`MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`

`FULL_PATCHMATCH = NOT_AUTHORIZED`

`LONGER_P_TRAINING = NOT_NEXT`

`CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`
