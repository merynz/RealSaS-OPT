# RealSaS-OPT — Current State

**Date:** 2026-08-27  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_CAMERA_SCALE_APPARATUS_CORRECTED__CALIBRATION8_V1_3_NEXT__V1_2_INVALIDATED__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question remains the **single-pose observable common-frame geometry hard tail**. E0 must separate:

1. observable-surface information loss (`E0-0 -> E0-a`);
2. deterministic persistence / SurfaceBuilder loss (`E0-a -> E0-b`);
3. downstream pretrained-consumer OOD, evaluated only after the primary scratch sufficiency decision.

No E0 PASS/FAIL has been declared. Proxy32 is closed.

## Frozen prior evidence — D2→D5

Frozen R256 FIT scale ladder, same architecture/objective:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

D2–D5 localized representative gauge-safe failures **before or at cross-view correspondence evidence/ranking**. Exact-normal plane warp and oracle visibility did not close the hard tail; coarse-to-fine search pruned truth too early. Search cannot recover truth that the evidence has already ranked away. This is not yet proof that a pretrained prior is required.

Canonical forensic report:
`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/P_HARDTAIL_ORACLE_D2_D5_20260826.md`

## E0 geometry arms — frozen

```text
E0-0  FULL-MESH CANONICAL CEILING
      deterministic area-uniform full source surface
      same RealSaS canonical/object frame
      matched point budget

E0-a  OBSERVABLE + ORACLE PERSISTENCE
      exact A×8 visible common-frame P
      P-only anchor selection
      teacher (triangle,bary) attached only after anchor freeze

E0-b  OBSERVABLE + DETERMINISTIC PERSISTENCE
      same exact visible P / same anchor budget as E0-a
      P-derived local normals
      frozen yaw/right/up orientation
      observable orthographic scale recovery
      common-frame proximity + reciprocal reprojection/cycle
      NO teacher identity in matching API
```

Primary causal interpretation:

```text
E0-0 -> E0-a = observable coverage / supported-surface gap
E0-a -> E0-b = deterministic persistence / SurfaceBuilder gap
```

The primary E0 information gate stays in the same canonical coordinate gauge. RigAnything/TokenRig-specific centering/scaling remains a later consumer-OOD diagnostic, not the substrate sufficiency definition.

## Calibration V1.2 — INVALIDATED APPARATUS WITNESS

`E0_CALIBRATION8_GEOMETRY_V1_2` completed with optimizer steps `0` and Proxy32 closed, but its scientific metrics are **not authority**.

Trigger: frozen calibration member `asset_76313e4bd82b82fcd1659c70` produced only `15` E0-a oracle-supported cross-view pairs. Because E0-a is the perfect-persistence upper bound, the run was treated fail-closed as an apparatus failure.

Root cause: the implementation projected every family with fixed orthographic half-extent `0.54`; the affected raster authority was rendered at a non-default scale. Post-hoc sidecar audit showed V0 half-extent `0.6172158837318421`.

No family was removed and no scientific acceptance threshold was changed. V1.2 cannot freeze downstream margins or support E0 PASS/FAIL.

Canonical correction report:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/E0_CAMERA_SCALE_APPARATUS_CORRECTION_V1.md`

## Corrected camera-scale authority

`camera.json` remains forbidden in E0 forward construction.

Per-view orthographic half-extent is directly recovered from already-admitted exact observable common-frame `P`, raster pixel-center coordinates, and frozen yaw/right/up:

```text
g_x = dot(P, right) / h
g_y = -dot(P, up) / h
```

A robust median of valid positive ratios estimates `h`; native-pixel reprojection residual is then checked fail-closed. `0.54` is only a historical default/test value, not runtime authority.

### Real frozen-family correction smoke

Patched source on real `asset_76313e4bd82b82fcd1659c70`, 512 anchors:

- recovered half-extent V0..V7: `0.6172158718109131`;
- camera recovery reprojection P95 max: `3.0517578125e-05 px`;
- E0-a oracle-supported pairs: `1175` (invalid V1.2: `15`);
- E0-b precision: `0.8950715421303657`;
- E0-b recall: `0.9582978723404255`;
- E0-b F1: `0.9256062474311549`;
- E0-b P-error P95: `0.002721910597756505`;
- `camera_json_consumed = false`;
- `teacher_identity_used_by_e0_b = false`;
- scientific optimizer steps = `0`.

Focused corrected source tests: `9/9 PASS`.

Geometry authority commit containing the correction:
`f5949e05484e2635cb36286312e6019e2293770a`

Calibration runner source snapshot:
`334737fca6ec78e63675b0788ad3377b3a96c9d8`

## V1.3 — CURRENT EXECUTABLE PACKAGE

Canonical package manifest:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/E0_CALIBRATION8_GEOMETRY_PACKAGE_V1.json`

Notebook:
`RealSaS_E0_CALIBRATION8_GEOMETRY_V1_3.ipynb`

SHA-256:
`06f613e95557263ff4f0f6c7761341d728553ba45da70bbd5df4c44821024b6a`

V1.3 preflight:

- all code cells compile;
- fresh embedded module path assertion PASS;
- Drive Errno103 retry self-test PASS;
- 8/8 calibration staging path PASS;
- focused unit tests `9/9 PASS`;
- real non-default-scale 763 apparatus witness PASS;
- independent real 763 full-density 512-anchor smoke PASS;
- Drive export hash self-test PASS;
- `camera.json` not staged/consumed;
- Proxy32 execution path absent;
- scientific optimizer steps `0`.

## Frozen population / next action

Membership is unchanged from the prospective FIT ladder:

- 512 FIT train-order authority for later fixed downstream probes;
- historical disjoint 8-asset FIT calibration panel;
- frozen family-disjoint `FIT_PROXY32` qualification panel;
- TUNE/CAL/DEV/EXTERNAL closed.

**NEXT EXECUTABLE STEP:** rerun exactly the same frozen Calibration-8 geometry panel using **V1.3**. Inspect the corrected geometry/persistence result before freezing any downstream scratch non-inferiority margins.

After corrected calibration geometry only:

1. inspect E0-0/E0-a/E0-b diagnostics;
2. freeze the scratch downstream probe and non-inferiority margins;
3. run matched scratch E0-0/E0-a/E0-b calibration;
4. only then may a separate Proxy32 E0 qualification package be authorized.

## AFTER E0 ONLY — correspondence intervention

Exactly three matched arms remain frozen:

```text
C1  scratch + current objective
C2  scratch + tail-aware correspondence objective
C3  pretrained visual/multiview prior transplant
    (MapAnything-family prior candidate; NOT the full camera/ray/product wrapper)
```

Only if representation demonstrates correct correspondence truth ranking/top-k containment may multiview search / fusion / propagation / PatchMatch-style mechanisms be promoted.

## Authorization state

`D2_D5_CORRESPONDENCE_LOCALIZATION = COMPLETE`

`E0_CALIBRATION8_V1_2 = INVALIDATED_APPARATUS_FIXED_HALF_EXTENT`

`E0_OBSERVABLE_CAMERA_SCALE_RECOVERY = REAL_763_512_SMOKE_PASS`

`E0_CORRECTED_SOURCE_TESTS = PASS_9_OF_9`

`E0_CALIBRATION8_V1_3 = NEXT`

`E0_PROXY32_QUALIFICATION = CLOSED`

`E0_SCIENTIFIC_OUTCOME = NOT_YET_OPENED`

`DOWNSTREAM_NONINFERIORITY_MARGINS = NOT_FROZEN`

`C1_SCRATCH_CURRENT = NOT_YET_EXECUTED`

`C2_SCRATCH_TAIL_AWARE = NOT_YET_EXECUTED`

`C3_PRETRAINED_PRIOR_TRANSPLANT = NOT_YET_EXECUTED`

`MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`

`FULL_PATCHMATCH = NOT_AUTHORIZED`

`LONGER_P_TRAINING = NOT_NEXT`

`TUNE_CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`
