# Geppetto/Arachne V0.1 — Character / Render Purity Audit Prereg V1

**Date:** 2026-08-31  
**Status:** `FROZEN_BEFORE_CHARACTER_RENDER_MEASUREMENTS`  
**Scope:** measurement-only input-quality audit; no learned-consumer output, no optimizer step, no quality threshold, no semantic exclusion.

## Question

The completed skeleton-projection audit established that the current FIT teacher geometry/rig/skin arrays satisfy the frozen structural contract. It did **not** establish that the selected assets are visually clean product-like characters.

Before Geppetto C0 optimizer steps, measure how much of the structurally admitted FIT population contains render/framing/geometry pathologies such as:

- tiny or severely cropped foreground;
- foreground touching the frame;
- fragmented/multi-object silhouettes;
- raster/image disagreement;
- one giant face/quad/plane dominating mesh surface or visible pixels;
- visible geometry with little/no skin support (diagnostic only);
- other objective tails that should inform a later render-usability freeze.

This audit deliberately does **not** decide whether an asset is semantically a character. Character-vs-prop classification is a later separately preregistered gate.

## Frozen population

Input authority is the already completed projection audit plus the frozen structural C0 policy:

- projection audit status: `PASS_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT`
- structural Geppetto C0 max controls: `160`
- overflow policy: `ABSTAIN_NO_TRUNCATION`
- expected structural Geppetto C0 assets: `2897`
- expected nested structural Arachne C0 assets: `2527`
- expected structural membership SHA-256: `cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147`

Only this FIT structural-C0 population may be measured. TUNE/CAL/DEV/EXTERNAL_HOLDOUT are not opened by this audit.

## Frozen data authority

For each admitted asset:

- geometry: `master/assets/<asset>/primary_geometry.npz`
- views: `master/assets/<asset>/renders/V0..V7/`
- raster: `raster_authority.npz`
- controlled image: `cel_clean_512.png`
- camera: `camera.json`

The eight cameras must retain the controlled yaw sequence `0,45,...,315` degrees. The audit does not rerender, repair, crop, normalize, delete, or mutate any source asset.

## Frozen objective measurements

### Per mesh

Using `vertices` and `faces`:

1. vertex count;
2. face count;
3. total surface area;
4. largest single-face fraction of total surface area;
5. top-two-face fraction of total surface area;
6. mesh bounding-box extents and nonzero-axis aspect diagnostic.

These measurements are descriptive. No giant-plane threshold is selected in this preregistration.

### Per view — raster geometry

Using `pixel_linear_index`, `triangle_id`, and `resolution` from `raster_authority.npz`:

1. foreground pixel count and occupancy fraction;
2. foreground bounding box width/height/area fraction;
3. minimum frame margin and border-touch flag;
4. foreground centroid offset;
5. 8-connected foreground component count;
6. significant foreground component count;
7. largest connected-component fraction;
8. visible triangle count;
9. largest visible-triangle pixel fraction;
10. top-two visible-triangle pixel fraction.

Malformed resolution, duplicate raster pixel IDs, out-of-range pixel IDs, or out-of-range triangle IDs are hard authority failures.

### Per view — controlled image consistency

`cel_clean_512.png` is compared to raster foreground without using learned segmentation.

Foreground source:
- if file alpha contains transparency: `alpha >= 128`;
- otherwise: complement of exact RGB green `(0,255,0)`.

Measured:
1. raster/image foreground IoU;
2. image foreground not supported by raster;
3. raster foreground missing from image.

No IoU threshold is chosen here.

### Skin-supported visibility — diagnostic only

If the `skin` vertex axis is compatible with the mesh, compute nonnegative per-vertex skin mass and project it through visible raster triangles to measure:

- mean visible triangle skin-support fraction;
- fraction of visible pixels lying on fully unskinned triangles.

This channel is **diagnostic only**. It is not valid as a Geppetto inference feature and cannot by itself exclude an asset in this measurement audit.

## Extreme-example registry

The runner records deterministic top-tail examples for later inspection, including:

- highest mesh face dominance;
- highest visible-triangle dominance;
- lowest frame margin;
- lowest raster/image IoU;
- highest component count;
- lowest/highest occupancy;
- highest visible unskinned fraction where valid.

Ranking is measurement-only and creates no membership decision.

## Hard failures

The full audit is `FAIL_HARD_RENDER_AUTHORITY_CONTRACT` if any selected structural-C0 asset cannot be measured under the frozen authority because of missing/malformed required geometry, raster, image, or camera evidence.

A truncated `--max-assets` run is `SMOKE_ONLY__NOT_FREEZEABLE`.

A complete zero-hard-failure run is:

`PASS_MEASUREMENT_COMPLETE__QUALITY_AND_SEMANTIC_FREEZE_NEXT`

A PASS authorizes **no training**.

## Leakage / decision firewall

This audit may not:

- read Geppetto, Arachne, IRIS, or Compiler prediction outputs;
- inspect TUNE/CAL/DEV/EXTERNAL_HOLDOUT;
- choose a render-quality threshold;
- classify character vs prop;
- mutate or rerender corpus assets;
- take optimizer steps;
- select a checkpoint.

## Next gate

After the complete result is sealed, and before model outputs are opened:

1. freeze objective render-usability thresholds from these input-quality measurements;
2. preregister a separate semantic `single riggable character` protocol that retains humanoids, creatures, quadrupeds, robots, monsters, and stylized characters while rejecting prop-only / environment / unrelated-scene assets;
3. intersect structural eligibility with render/character eligibility to produce final C0 training membership;
4. only then build and run the Geppetto zero-step training apparatus.
