# E0 Camera-Scale Apparatus Correction V1

**Date:** 2026-08-27  
**Branch:** `g0-g1/single-pose-geometry`  
**Decision:** `CALIBRATION_V1_2_INVALIDATED__APPARATUS_CORRECTION_REQUIRED__PROXY32_CLOSED`

## Trigger

The sealed Calibration-8 V1.2 run completed with scientific optimizer steps `0` and Proxy32 closed, but `asset_76313e4bd82b82fcd1659c70` produced only `15` E0-a oracle-supported cross-view pairs. Because E0-a is the perfect-persistence upper-bound arm, this was treated as an apparatus failure, not as a SurfaceBuilder result.

## Root cause

The E0 projection helper used fixed orthographic half-extent `0.54` for every family. Post-hoc inspection showed the affected asset's V0 raster was rendered at half-extent `0.6172158837318421`. The fixed-scale projection therefore looked for the correct physical carrier at the wrong raster location and collapsed the oracle witness search.

No calibration member is removed and no scientific acceptance threshold is changed.

## Leak-safe correction

`camera.json` remains forbidden as a forward E0 input. Orthographic half-extent is directly recoverable from already-admitted exact observable common-frame `P`, raster pixel-center coordinates, and frozen yaw/right/up orientation:

```text
g_x = dot(P, right) / h
g_y = -dot(P, up) / h
```

The corrected apparatus forms valid positive `h` ratios from both screen axes, takes a robust median, then fail-closes on native-pixel reprojection residual. This changes camera scale authority only; E0 targets, membership, point budget, persistence gates, and scientific optimizer count are unchanged.

## Real-family closure smoke

Patched source was run on the real frozen `asset_76313e4bd82b82fcd1659c70` authority (`primary_geometry.npz` + V0..V7 native raster authority), 512 anchors:

- recovered half-extent, V0..V7: `0.6172158718109131` each;
- recovered-scale reprojection P95, V0..V7: `3.0517578125e-05 px` each;
- E0-a oracle-supported pairs: `1175` (invalid V1.2: `15`);
- E0-b predicted pairs: `1258`;
- E0-b precision: `0.8950715421303657`;
- E0-b recall: `0.9582978723404255`;
- E0-b F1: `0.9256062474311549`;
- E0-b common-frame P-error P95: `0.002721910597756505`;
- `camera_json_consumed = false`;
- `teacher_identity_used_by_e0_b = false`;
- scientific optimizer steps: `0`.

Unit/source closure after correction: `9/9 PASS` for the focused surface-builder suite, including a non-default half-extent recovery test.

## Scientific consequence

Calibration V1.2 is retained only as an invalidated apparatus witness. Its aggregate or per-family geometry metrics must not freeze downstream non-inferiority margins and must not support E0 PASS/FAIL.

The next authorized action is exactly one rerun of the same frozen Calibration-8 membership under the corrected observable-scale apparatus. Proxy32, C1/C2/C3, PatchMatch/search/fusion/propagation, and pretrained downstream primary gating remain closed.
