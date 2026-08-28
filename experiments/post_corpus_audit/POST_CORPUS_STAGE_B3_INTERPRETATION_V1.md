# RealSaS Post-Corpus Stage-B3 Interpretation V1

**Date:** 2026-08-23

## Exhaustive `.blend` authority result

Stage-B3 audited all 385 selected `.blend` assets with the Stage-B2 bidirectional physical-surface metric.

- 385/385 probes succeeded; failures: 0.
- evaluated-surface p95 > 1e-4: 55 assets.
- evaluated-surface p95 > 5e-4: 51 assets.
- evaluated-surface p95 > 1e-3: 50 assets.
- evaluated topology changes: 57 assets.
- non-ARMATURE modifier assets: 114.
- active non-Basis shape-key assets: 6.
- triangulation-surface p95 > 5e-4: 9 assets; >1e-3: 7.

The diagnostic classifier is decisive: every material evaluated-surface failure (51/51) is captured by `non-ARMATURE modifier OR active non-Basis shape key`; missed material failures = 0.

## Repair set

`BLEND_GEOMETRY_REPAIR_CANDIDATES_V1.json` contains 62 assets, all from `objaverse_animated_originals`.

Reason union:

- 47 evaluated-surface only;
- 7 triangulation-surface only;
- 4 active-shape qualification only;
- 2 evaluated + active-shape;
- 2 evaluated + triangulation.

Dominant non-ARMATURE modifiers among the 62: SUBSURF 26, MIRROR 24, DECIMATE 7, BEVEL 4, SOLIDIFY 4, NODES 3, with a small tail of COLLISION/EDGE_SPLIT/SMOOTH/SHRINKWRAP/MASK/MULTIRES.

## Decision

Gate-2 is no longer an unknown: the current base-mesh/fan extraction is materially wrong for a bounded subset. A bounded geometry repair + rerender is required; a global 3993 rerender is not justified by Gate-2 alone.

Before mutation, Stage-B4 must prove that evaluated topology preserves usable rig/skinning authority for currently Arachne-capable candidates. If evaluated deform-vertex groups are preserved and satisfy the existing <=1% zero-row technical threshold, repair may use evaluated rest geometry + Blender loop triangles + propagated skin weights. Any failures require explicit deterministic transfer or capability downgrade rather than silent corruption.

Appearance Gate-3 remains separate: source dependency closure must be recovered before appearance-preserving B renders.
