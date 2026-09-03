# RealSaS — Observation-Derived Appearance Pixel-Center Behavioral Closure — 2026-09-03

**Status:** `PASS_APPEARANCE_PIXEL_CENTER_BEHAVIORAL_CLOSED`

## Boundary

Behavioral authority for:

`Qualified directional mesh + admitted surface raster provenance -> observed per-corner AppearanceBindingIR`

This closure covers coordinate/provenance semantics only. It does not claim visual completion quality or generalization.

## Frozen preregistration

- `canonical/APPEARANCE_PIXEL_CENTER_BEHAVIORAL_PREREG_20260903.md`
- `compiler/test_appearance_pixel_center_behavioral_v1.py`
- `.github/workflows/appearance_pixel_center_behavioral_v1.yml`

The behavioral witness and PASS criteria were frozen before source repair.

## Baseline falsification

Baseline workflow:

- run `33782587984`
- job `100739599575`

Observed failure on native `PIXEL_CENTER_XY` input:

- donor pixel x = `255.5`
- expected normalized material u = `(255.5 + 0.5) / 1024 = 0.25`
- shipping producer emitted `128.25`
- absolute error = `128.0`

The camera-mismatch fail-closed test passed in the same baseline. Therefore the defect was isolated to raster-coordinate interpretation, not camera lineage enforcement.

## Generic repair

`compiler/realsas_compiler_core/appearance.py` now:

- requires `surface.metadata.raster_coordinate_system == PIXEL_CENTER_XY`;
- requires a positive native raster `resolution`;
- preserves `donor_raster_xy` in native pixel-center units;
- converts to normalized material UV by the exact inverse pixel-center convention:

  `u = (x + 0.5) / resolution`

  `v = 1 - (y + 0.5) / resolution`

- rejects non-finite or materially out-of-frame donor raster coordinates;
- includes raster coordinate system, resolution and UV convention in appearance provenance metadata;
- keeps `camera_refit=false`, `source_mesh_uv_used=false`, and `unknown_completion_used=false`;
- adds no legacy normalized-grid fallback.

Synthetic legacy fixtures were migrated to the production coordinate contract rather than weakening the product producer.

## Repaired PASS evidence

Frozen behavioral workflow:

- run `33783081145`
- job `100741218452`
- `2 passed`

The frozen gate verifies:

- exact face-corner coverage;
- local donor authority;
- exact donor pixel-center preservation;
- exact normalized UV conversion;
- UV remains within `[0,1]` for native in-frame witnesses;
- observation-hash mutation changes corner source hashes and appearance lineage without moving UV/raster coordinates;
- camera mismatch fails closed.

Additional regression evidence on the same repaired head:

- appearance source contract run `33783081113`: PASS;
- Architecture V4 contract run `33783081117`: PASS;
- COMPLETE_E2E source contract run `33783081131`: PASS through Compiler -> proof -> reference runtime.

## Closure verdict

`APPEARANCE_PIXEL_CENTER_BEHAVIORAL = PASS/CLOSED`

The former normalized-grid reinterpretation is forbidden for production observed appearance. The next behavioral seam is deterministic puppet-local motion: a preset must cause measurable canonical puppet state/deformation change, not merely contain a non-constant key track.
