# RealSaS — Observation-Derived Appearance Pixel-Center Behavioral Preregistration — 2026-09-03

**Status:** `FROZEN_BEFORE_BASELINE_RESULT`

## Seam

`Qualified directional MWB2 mesh + RiggingSurfaceIR raster provenance -> build_observed_appearance_binding -> AppearanceBindingIR`

This gate tests coordinate/provenance semantics only. It makes no perceptual-completion or generalization claim.

## Authoritative input contract

Production IRIS V2 emits and persistence preserves:

- `raster_coordinate_system = PIXEL_CENTER_XY`;
- `resolution = 1024`;
- per-view raster coordinates in native pixel-center units;
- exact observation/camera lineage.

No source mesh UV, camera refit, teacher UV, resized raster, or normalized-grid reinterpretation is allowed.

## Frozen witness

A generic four-node observed raster cell is used:

- native resolution: `1024`;
- target view: `0`;
- pixel-center coordinates chosen strictly inside the native frame;
- all six K4 local relations are safe so repaired MWB2 deterministically produces exactly two non-overlapping faces;
- mesh vertices are identity-bound to admitted surface nodes.

No real family values or fitted thresholds are present.

## Frozen PASS criteria

For every mesh face corner:

1. appearance coverage is exact: one binding for every `(face_index, corner_index)` and no extras;
2. authority is `OBSERVED_LOCAL` and donor view equals target view;
3. donor raster coordinate equals the admitted surface node's native `PIXEL_CENTER_XY` coordinate exactly;
4. material UV is the exact pixel-center normalization:

   `u = (x + 0.5) / 1024`

   `v = 1 - (y + 0.5) / 1024`

5. both UV coordinates lie in `[0,1]`;
6. `camera_refit = false`, `source_mesh_uv_used = false`, `unknown_completion_used = false`;
7. changing only the donor observation hash changes every affected corner source-observation hash and the appearance lineage hash while leaving donor coordinates/UV unchanged;
8. a mismatched camera binding fails closed.

## Failure policy

If baseline fails, the witness, coordinates, formula and thresholds are frozen. Repair must be generic to the declared raster coordinate system/resolution and may not special-case this witness.

## Authorization

Source repair is authorized only after the frozen baseline result is observed. Architecture refreeze remains blocked until this seam passes.
