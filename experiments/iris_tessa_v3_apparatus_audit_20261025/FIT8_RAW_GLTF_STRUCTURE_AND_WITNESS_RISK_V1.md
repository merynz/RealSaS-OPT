# FIT8 Raw glTF Structure & Witness-Risk Census V1

**Audit date:** 2026-10-25  
**Authority:** read-only exact-source census; no import/render admission, no product/training authority.

## Purpose

Ground the V3 witness-sampling and Truth Foundry ontology audit in exact authored FIT8 source bytes rather than abstract examples.

Four FIT8 subjects use glTF source assets that can be parsed directly without Blender. Their downloaded bytes were SHA256-verified against `canonical/FIT8_DIVERSITY_SELECTION_V1.json` before this census.

The other four FIT8 subjects are FBX and are not silently approximated here.

## Exact source census

### Bat — flying

Canonical source SHA256: `47574f68f8c9e1c953a09f79e4b5f5fb1c07e240130d208fbd63bd91d0ecb63c`

- mesh objects: 1
- primitives: 1
- primitive POSITION vertices: 1,008
- triangles: 1,200
- materials: 1
- nodes: 17
- skins: 1
- joints: 15
- native animations: 4

The single mesh/material authoring structure still contains articulated wing/body behavior; material or mesh identity cannot stand in for final edit/drawable ontology.

### Horse — quadruped

Canonical source SHA256: `3deb61550dff1d2786d04b6e8559d63ad3907d6ab606ba28ce0af074ed96341b`

- mesh objects: 1
- primitives: 8
- total primitive POSITION vertices: 4,400
- total triangles: 2,182
- materials: 8
- nodes: 52
- skins: 1
- joints: 50
- native animations: 13

Primitive detail:

| primitive | vertices | triangles | material |
|---:|---:|---:|---:|
| 0 | 3,086 | 1,532 | 0 |
| 1 | 124 | 60 | 1 |
| 2 | 304 | 152 | 2 |
| 3 | 180 | 88 | 3 |
| 4 | 510 | 254 | 4 |
| 5 | 98 | 50 | 5 |
| 6 | 14 | 6 | 6 |
| 7 | 84 | 40 | 7 |

This is a strong source-structure imbalance case: the smallest authored primitive has only 14 vertices while the largest has 3,086.

### GreenBlob — amorphous

Canonical source SHA256: `e2d2049ebe8d243bd0f60a3f48a7e174444f641ebc40cbea2f6c23cb97a82519`

- mesh objects: 2
- primitives: 2
- total primitive POSITION vertices: 878
- total triangles: 1,520
- materials: 1
- nodes: 7
- skins: 1
- joints: 4
- native animations: 9

Primitive detail:

| mesh | primitive | vertices | triangles | material |
|---:|---:|---:|---:|---:|
| 0 | 0 | 661 | 1,152 | 0 |
| 1 | 0 | 217 | 368 | 0 |

Two authored mesh nodes share one material; material identity is therefore not equivalent to mesh/source support.

### George — mechanical

Canonical source SHA256: `eccc45d82701f32d67b4480346757b1656c649d9d40e4a1484f57d660596285e`

- mesh objects: 1
- primitives: 1
- primitive POSITION vertices: 7,755
- triangles: 7,864
- materials: 1
- nodes: 49
- skins: 1
- joints: 47
- native animations: 20

George is particularly important for the role-free hypothesis: a single mesh / primitive / material source carries a highly articulated mechanical system. Source object/material decomposition therefore cannot supply the desired graphics ontology.

## Raw-concatenation index-sampler proxy

The current V3 witness sampler does **not** operate on raw glTF vertices. It operates on the temporary FlexiCubes extracted surface and chooses indices by:

`floor(i * extracted_vertex_count / 1024)`.

Therefore the following is explicitly a **source-structure proxy only**, not a claimed product/extracted-surface result.

If each asset's raw primitive vertex arrays were concatenated in the order above and the exact same 1,024-index rule were applied:

- Bat: one primitive -> 1,024 witnesses;
- GreenBlob: approximately 771 / 253 witnesses across its two primitives;
- George: one primitive -> 1,024 witnesses;
- Horse: approximately `[719, 29, 70, 42, 119, 23, 3, 19]` witnesses across primitives 0..7.

Thus the Horse source proxy does **not** demonstrate an actual omitted primitive, but it demonstrates extreme stratum imbalance: the 14-vertex primitive receives only ~3 of 1,024 samples under a simple contiguous-order proxy.

## What is proven vs open

Proven from source/code:

1. FIT8 authoring decomposition is heterogeneous enough that mesh/material/primitive IDs cannot be promoted into a generic graphics ontology.
2. Current V3 `deterministic_surface_witness_v1` has no component, area, boundary, attachment, occlusion or thinness coverage guarantee.
3. Real source assets contain strongly imbalanced structural strata (Horse is an exact example).

Not proven:

- that the current FlexiCubes extracted surface preserves source primitive contiguity;
- that any actual FIT8 extracted component is missed by the 1,024 witnesses;
- that 1,024 is intrinsically too small.

## Required closure

Before changing witness count or policy, run an exact extracted-surface coverage court on the frozen candidate IRIS apparatus. For each subject report at minimum:

- extracted vertex/face count;
- connected-component coverage;
- surface-area coverage;
- boundary/curvature/thinness strata;
- causal attachment/contact strata when Truth Foundry supplies them;
- occlusion/grazing strata;
- minimum witness support per qualified stratum.

If material gaps exist, compare the current index sampler against a deterministic coverage-aware sampler under the same witness budget before increasing `surface_witness_count`.

No source primitive/material label should be exposed to the student as a semantic target merely to repair sampling.
