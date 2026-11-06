# Corpus370 — Original Triceratops physical palette V1 FAIL and diagonal subpixel source proof

2026-11-06 · PR #351 research. **NO GRAPHICS OWNER OR TRAINING AUTHORITY.**

The original Triceratops FBX (SHA ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a) produced 16 real physical-color-edit Blender renders: 4 original camera views V0/V2/V4/V6, each baseline and 3 independent palette color edits. 256px, Cycles CPU, fixed seed, 16 samples, source unlit/alpha retained policy.

Independent original Blender V1 ray-locality court **FAIL**. Edit palette index2 unexpectedly changed one ostensibly unselected ray interior in V6 at top-left x166, y134; RGB max difference 0.1764706075 exceeds the unchanged fixed 3/255 gate, alpha before/after 1.0. No selected interior failed to change; no other sampled unintended change. Per color (selected/unselected/collateral): 0=13/1461/0, 1=1417/57/0, 2=44/1430/1. This is NOT PASS and was preserved privately with full original source render SHAs.

- [V1 independent failed source ray court](https://drive.google.com/file/d/19b6sRZn9WucrFxlupztuHApWrUsmgP3d/view)
- [V1 original physical edit and recovery workflow](https://github.com/merynz/RealSaS-OPT/actions/runs/42238824415)

## Second independent original-source geometric falsification

The exact original source rest NPZ (SHA 282cdd2813da95ab65ec8211c8338daeb263395659f556f2e7c01d76cc9afcd0; 669 vertices / 1334 triangles), independently Blender-verified face palette facts (SHA 8ff5534e4e8d7549cb9d419e4aea58511e0e042f9113e7d5807a8652a1b76edc) and original camera bundle (SHA e73917f650ee2dd925efe68a88e2fafdd5f99dbc5b7c410b7de0532ecf66bb0d) were intersected using a NEW, separate vectorized analytic Möller–Trumbore nearest-positive-depth ray implementation, not Blender scene.ray_cast or the edit renderer.

At V6 pixel x166,y134, the source center and all cardinal x/y offset rays up to ±1.75px see original palette1. Yet a diagonal ray dx=-0.25px,dy=+0.25px sees source palette2. An 11×11 subpixel source-ray grid includes both palette1 and palette2.

**Consequence:** center/cardinal first-hit consistency is NOT sufficient to certify a uniform pixel-footprint palette. The renderer's palette2 edit can change a pixel even when its center ray and cardinal probes see palette1. Exact Cycles sample positions were NOT reconstructed, so the causal statement remains source-geometry-consistent, not fully sample-exact.

This invalidates the V1 interior predicate. V1 **remains FAIL** and its single unexpected pixel is NOT yet established as true artist-owner collateral leakage.

## Preregistered V2

[Original subpixel footprint V2 preregistration](CORPUS370_ORIGINAL_PALETTE_PHYSICAL_FOOTPRINT_V2_PREREG_20261106.md) introduces an independent source-rest analytic 3×3 pixel subray stencil at offsets ±0.375,0; any mixed/unknown pixel remains UNKNOWN. It retains all observed-pixel tolerances, sample grid stride, fixed renderer and exact color edit policy. It tests **different physical edit camera directions V1/V3/V5/V7**, not the inspected V0/V2/V4/V6 set. These are held out relative to physical edit consequences, not globally unseen subjects or poses.

This is a bounded scene/world/sample coverage result, **not full continuous-pixel area proof**, raw lit shader parity, independently editable drawable ownership, pose2, attachments, motion, collision, generic unseen morphology or any IRIS/TESSA training authority.

All training/materialization/graphics owner flags remain FALSE regardless of V2 result.
