# Post-Corpus Stage-B4 Interpretation V1

**Date:** 2026-08-23

## Raw result

- repair candidates: 62/62 probed successfully
- current capabilities among those candidates: IRIS 62, Geppetto 62, Arachne 45
- topology-changed candidates: 48
- topology-changed objects with zero evaluated group elements: 119
- current Arachne-capable candidates failing the post-evaluated <=1% zero-row gate: 1 (`asset_d63f57516add9e9f9e81a87d`, 4.8385%)

## Critical interpretation correction

The raw `119` count is **not evidence that Blender lost skin weights during evaluated-mesh generation**.

Full result analysis shows:

- 119/119 topology-changed objects with zero evaluated group elements also had zero base group elements;
- 0 topology-changed objects had base group elements >0 and evaluated group elements ==0;
- 55 topology-changed objects with base group elements >0 also retained evaluated group elements;
- per-object zero-row fraction on topology-changed objects never worsened (some improved slightly).

Therefore `to_mesh(preserve_all_data_layers=True, depsgraph=deps)` is adequate for the existing weighted objects in this repair set. No synthetic nearest-neighbour or barycentric skin-transfer algorithm is authorized.

The sole global Arachne downgrade candidate, `asset_d63f57516add9e9f9e81a87d`, is different: it already contains unweighted decorative objects in the base source. `BEVEL` increases the vertex count of those still-unweighted components, changing the global vertex-count-based zero-row fraction from ~0.80% to ~4.84%. This is a teacher-coverage issue, not a propagation failure. Do not invent weights; repair geometry, then downgrade Arachne capability if the canonical technical gate still fails.

## Shape-key authority

Six repair candidates contain active non-Basis shape keys. These include expression/morph-like keys such as `eye blink`, `mouth_d+`, `mouth_d-`, `Feet`, `Hands`, plus source-specific keys. Automatic baking would silently choose a morph/animation state as canonical rest geometry.

Policy: **quarantine these six from the automatic repair/training path until separately qualified.** This costs negligible corpus scale and preserves target authority.

## Authorized next step

Stage-B5 is a read-only repair dry run on the remaining 56 candidates:

- evaluated REST geometry;
- Blender loop-triangle authority;
- preserved evaluated deform layers;
- V4.3 canonical normalization and technical gates;
- no Drive asset overwrite;
- no render;
- no training.

Only after Stage-B5 passes may repaired geometry be published and affected A-pass renders be regenerated.
