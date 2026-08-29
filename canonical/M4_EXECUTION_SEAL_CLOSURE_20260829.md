# RealSaS — M4 Pre-open Execution Seal Closure

**Date:** 2026-08-29  
**Status:** `SEALED_BEFORE_M4_SCIENTIFIC_OUTCOMES`

This closure binds the preregistered M4 structured forward-depth grid to exact raw-source, runner, runtime, project-local dependency, typed Compiler-adapter, frozen proxy-target and D2 checkpoint authorities before any M4 scientific cell is opened.

## Population and raw source authority

The population is the frozen eight-family `calibration_anchor8` set. For each family the canonical master source contributes exactly one `primary_geometry.npz` and eight native-1024 `raster_authority.npz` files.

- assets: 8
- raw source files: 72/72
- ZIP integrity: PASS
- NPZ load integrity: PASS
- raw-source manifest SHA-256: `f044587a12b51a8ae141bb6a3b37c67b7120f9e391e0f0d2afebad939a2da7c8`

Duplicate run-output copies were not admitted as source authority. Canonical sources resolve through `RealSaS_MASTER_CORPUS_1024_V3/master/assets`.

## Frozen execution bytes

- base main commit: `eb18e8a0e12d813419e6e8274c29f784c560daaf`
- M4 prereg JSON SHA-256: `c23166ff45a9c12fa9978f448a01e0501c5a1972aa5b46fd06bcef3d1305f0e8`
- grid runtime SHA-256: `4ebf3636438aa14b5f832d51da3e9222086b948b7480164dcd3c50d3ad4ca9dd`
- Compiler surface adapter SHA-256: `e81e2eb25145238d7dcc42b493ba94825ac202dd5eed31335537c33372e1b589`
- M4 grid runner SHA-256: `48462efcf80d6d16ef95dadaa1932f2aaed0f5e5849b0b2ef8961a005e31a667`
- Geppetto D2 checkpoint SHA-256: `f8c6146fc3ad81146ced01805b9be454ad194b86db3a9ab3d72d7b9eb3747b65`
- Arachne D2 checkpoint SHA-256: `72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745`

The pre-open transitive audit found two project-local bridge dependencies that the first branch draft did not pin explicitly: `bridge_persistence_v1.py` and `depth_corruption_v1.py`. It also found that a normal `realsas_compiler_core.surface` import executes unrelated package-init/vendor restoration. Before any scientific outcome and before promotion to `main`, the runner was corrected to:

1. pin every project-local M4 route dependency by current-main Git blob ID;
2. load the exact `surface.py + types.py + hashing.py` modules in an isolated package context, so no unrelated Compiler package-init side effects enter the M4 surface-adapter experiment;
3. pin all four frozen proxy target packs by SHA-256.

The four target pack SHAs were independently replayed against the historical E0 prep index and matched 4/4.

## Pre-open checks

- 65-cell frozen enumeration: 65/65 unique
- baseline: `E0000_L000_ALL8`
- eight-family zero-only asset preparation: PASS, 512 anchors/family, all 8 source views represented
- isolated exact typed Compiler adapter smoke: PASS, 512 `RiggingSurfaceIR` nodes
- frozen proxy target pack byte audit: 4/4 PASS
- project-local transitive dependency audit: PASS
- scientific optimizer steps: 0
- Proxy27: CLOSED
- DEV32: CLOSED
- M4 scientific outcomes opened: false

`run_m4_surface_grid_v1.py` is fail-closed. Its default mode performs preflight only. Scientific execution requires an explicit `--open-scientific-outcomes` flag plus a matching `M4_EXECUTION_SEAL_V1.json`; the runner rejects raw-source, project-byte, target-pack, checkpoint, population, grid, or seal drift before executing the grid.

## Authorization boundary

This seal authorizes opening the preregistered 65-cell M4 surface/substrate grid only after the corrected seal is promoted to canonical `main` and replayed there. It does not itself contain a tolerance outcome and does not authorize a product-safe IRIS claim. The consumer-validity interlock remains mandatory after the bridge result.
