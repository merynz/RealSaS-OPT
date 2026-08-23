# Post-Corpus Stage-B Interpretation V1

**Date:** 2026-08-23  
**Branch:** `g0-g1/single-pose-geometry`  
**Evidence:** `POST_CORPUS_STAGE_B_SELECTIVE_AUTHORITY_RESULT_V1.json` produced from 117/117 successful Blender source probes.

## Raw Stage-B headline

- unique sample sources: 117
- successful probes: 117/117
- `.blend` sample: 85
- old binary `evaluated_diff` flag: 63/85
- old fan-vs-Blender triangulation mismatch flag: 61/85
- any modifier: 83/85 `.blend`
- old shape-key flag: 5/85 `.blend`
- materials: 110/117
- image nodes: 53/117
- packed images: 17/117
- missing external image references: 67/117
- builder-vs-authored-corner normal asset-summary median: 6.71 deg; p90 37.46 deg

## Critical interpretation correction

The raw `63/85 evaluated geometry diff` number is **not** sufficient to declare 63 assets invalid. Stage-B V1 classified an evaluated mesh as different when indexwise world-coordinate delta exceeded `1e-7`, which is too sensitive for a product-surface decision.

Detailed parsing of the V1 result gives, within the 85 `.blend` sample:

- 63/85 trigger the old `>1e-7` evaluated-diff flag;
- only **8/85** change evaluated vertex and/or triangle counts;
- only **3/85** have an indexwise world-vertex max delta above `1e-6` when counts are equal;
- only **1/85** exceeds `1e-5`;
- median asset max delta is about `2.46e-7`, p95 about `9.42e-7`, max about `6.95e-3`;
- only **24/85** carry a viewport-enabled non-ARMATURE modifier type;
- the 83/85 any-modifier rate is dominated by the expected ARMATURE modifier and is not itself a defect.

Therefore Gate 2 remains **OPEN_FOR_CONSEQUENCE**, not FAIL.

## Triangulation interpretation

`61/85` fan-vs-Blender triangle-index mismatch is also not automatically a surface error. Convex or near-planar quads can choose a different diagonal while representing effectively the same physical surface. Exact source-topology mismatch is not a product failure unless it causes a material geometric/raster consequence.

Stage-B2 is therefore preregistered to measure bidirectional base-vs-evaluated and fan-vs-Blender **surface distance**, normalized by object bounding-box diagonal. It reports bands at `1e-5`, `1e-4`, `5e-4`, and `1e-3` rather than promoting an index mismatch to a corpus failure.

## Shape-key interpretation

Stage-B V1's shape-key count includes any key-block collection, including Basis-only state. Stage-B2 counts only active non-Basis keys with nonzero value before treating shape-key state as a geometry-authority concern.

## Normal authority correction

Authored/custom/split corner normals are shading authority, not automatically product geometric authority. For IRIS's rigging substrate, exact triangle+barycentric truth allows the teacher normal to be defined as a geometric face/surface normal directly from the canonical mesh. Source shading-normal disagreement will therefore be reported as an observation/shading distinction, not automatically a target-authority defect.

## Appearance finding

Appearance loss is material and cannot be dismissed:

- 110/117 sampled sources contain materials;
- 53/117 contain image texture nodes;
- 17/117 have packed images;
- 67/117 reference external images that are missing from the locally preserved single-file source context.

Breakdown from Stage-B V1 detailed result:

- `.blend`: 55/85 sampled assets have missing external images;
- `.fbx`: 12/24;
- `.glb`: 0/8.

This is consistent with V4.3's deliberate storage policy: reproducible linked raw objects are not all persisted, and single preserved files do not imply dependency closure. The appearance-preserving B observation pass must therefore materialize source **dependency closure/package context**, not merely reopen a lone `.blend`/`.fbx` file.

## Current decisions

1. Do **not** globally rerender yet based on Stage-B V1.
2. Run Stage-B2 surface-consequence audit on the same 117-source sample.
3. If material evaluated/fan-triangulation surface deviations are concentrated in a bounded subset, repair/re-extract/rerender only that affected class/subset.
4. If deviations are systemic at product/raster scale, revise extraction authority and rerender globally despite compute cost.
5. Independently, Gate 3 now requires dependency-aware appearance recovery before final IRIS training; A remains the geometry-isolation control and B/C are sibling observations.
