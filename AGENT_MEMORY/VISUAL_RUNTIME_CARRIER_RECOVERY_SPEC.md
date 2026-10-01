# Visual Runtime Carrier Recovery Specification

> NON-AUTHORITATIVE RECOVERY DESIGN. This defines the seam to repair; it does not declare any current visual deformation operator product-qualified.

## Problem statement

Normalized Stage38 correctly seals source-owned visual authority:

- `source_owned_visual_mesh_mode = true`
- `visual_mesh_set_binding_hash = VisualMeshSetIR.set_hash`
- `mechanical_mesh_render_authority = false`

Normalized Stage42 then correctly fails closed because the runtime contract has no typed representation for source-owned presentation geometry.

Current `RuntimeProjectionV2IR` and RSS V2 assume:

```text
one mechanical mesh
+ mechanical face UV
+ Stage41 posed 3D mechanical vertex frames
= render geometry
```

That assumption is invalid once Stage18/38 revoke mechanical mesh render authority.

## Critical architecture rule

**Transport authority and deformation-operator quality must be separated.**

The recovery must not hard-code a research operator merely because it is the latest experiment.

The 1941/1958 research history contains several visual-deformation candidates:
- global presentation-handle ARAP;
- direct mechanical barycentric binding + ARAP for unbound vertices;
- region-aware safe mechanical affine binding;
- source-owned deformation-region variants.

They are evidence for how to move visual geometry, but none is universally product-qualified.

Therefore:

```text
source-owned visual carrier = canonical platform contract
visual deformation operator = replaceable Stage42 implementation detail
Stage45 dynamic visual proof = shipping authority
```

## Proposed Stage42 product contract

Before Stage42, Stage37 must qualify the final source-owned presentation geometry using Stage35-qualified mechanical evidence. Stage18 visual output is substrate/candidate evidence, not automatically final shipping topology.

Stage42 then becomes a compiler/runtime-projection stage with two explicit geometry modes:

### MECHANICAL_PRESENTATION_V1

Legacy/current mechanical render path, only valid when the sealed product explicitly grants mechanical render authority.

### SOURCE_OWNED_VISUAL_PRESENTATION_V1

Required when Stage38 carries a non-empty `visual_mesh_set_binding_hash` and mechanical render authority is false.

In source-owned mode Stage42 must consume:

- Stage37 qualified presentation visual mesh/binding derived from source-owned visual substrate plus Stage35-qualified mechanics;
- Stage38 exact qualified visual-mesh/binding seal;
- Stage41 exact posed mechanical motion;
- Stage23/24 appearance authority;
- Stage05 cameras;
- the selected visual deformation implementation + policy identity.

It emits a typed runtime projection whose render geometry is the visual mesh, never the mechanical mesh.

## Runtime IR changes

The runtime projection needs first-class fields conceptually equivalent to:

```text
presentation_geometry_mode
visual_mesh_set_binding_hash
visual_deformation_operator_id
visual_deformation_policy_hash
visual_geometry_artifact_hash
```

Exact schema version may become `RuntimeProjectionIR.v3` rather than weakening V2 semantics.

For each view, runtime projection must bind:

- exact qualified presentation visual-view mesh hash;
- visual vertex count;
- visual face count;
- source dimensions;
- fixed source-raster UV;
- exact source texture identity;
- compiled per-clip/per-frame visual positions or another typed qualified visual-motion payload.

No implicit lookup of “latest” visual artifacts is allowed.

## RSS/package changes

The package must distinguish render geometry from mechanical simulation authority.

Conceptual entries:

```text
manifest
cameras

visual/V0/mesh
visual/V0/clip_0_positions
...
visual/V7/mesh
visual/V7/clip_N_positions

source textures / CAA payload
provenance

mechanical identity metadata (binding/proof only)
```

The native player renders visual geometry when `presentation_geometry_mode=SOURCE_OWNED_VISUAL_PRESENTATION_V1`.

The mechanical mesh remains bound for provenance/mechanics but may not be rasterized as a silent fallback.

## Runtime restrictions

In product/native playback:

- no model fit;
- no donor search;
- no appearance generation;
- no hidden retriangulation;
- no visual-mesh rebuild;
- no unsealed operator selection;
- no fallback to mechanical render mesh.

All expensive/experimental visual deformation work happens at compile/Stage42 and is sealed into typed artifacts. Playback is deterministic.

## Deformation operator boundary

Stage42 may compile visual positions using a selected operator implementation, but that operator is not product authority by itself.

A candidate operator must emit QA sufficient for Stage45, including at minimum:
- triangle orientation/flip evidence;
- intrinsic visual edge stretch;
- binding/support residual;
- source-mask exposure metrics;
- visibility/projection metrics;
- exact operator + policy identity.

Stage45 decides whether the resulting dynamic visual presentation is acceptable.

This lets the research loop be:

```text
change visual deformation operator
        ↓
invalidate Stage42–46 only
        ↓
reuse models / GSA / mesh / skeleton / skin / CAA / motion
        ↓
render and inspect in minutes
```

rather than replaying Stage01–41.

## Existing reusable primitives

Already in compiler core:
- `VisualMeshSetIR` and per-view immutable source-owned visual substrate meshes;
- `bind_source_visual_points_to_projected_surface_v1`;
- `bind_region_visual_vertices_to_mechanical_affine_v1`;
- `evaluate_region_visual_binding_v1`;
- ARAP2D implementation and QA helpers;
- source/raster coordinate transforms.

Already demonstrated in product research:
- direct render-only sealed artifact reuse;
- C++ visual-mesh rasterization;
- source texture fixed-UV rendering;
- visual→mechanical binding;
- mechanical-support counterfactual proving catastrophic renderer-support mismatch.

The recovery should promote generic primitives into the typed runtime seam, not import Knight-specific demo code.

## Explicit non-goals

This repair does not:
- declare region-affine, ARAP or another current operator “final”;
- solve residual dynamic visual exposure by fiat;
- move research scripts into production wholesale;
- restore old IRIS research implementations;
- change mechanical authority.

## Acceptance tests

1. Stage38 source-owned mode cannot reach Stage42 without exact VisualMeshSet identity.
2. Stage42 source-owned mode emits no mechanical `faces+face_uv` render authority.
3. RSS contains typed source-owned visual geometry for every admitted view.
4. Native playback consumes that visual geometry and has no mechanical fallback path.
5. Product render path executes zero model-fit stages.
6. Changing only the visual deformation implementation invalidates Stage42–46, not Stage01–41.
7. Repeating render against the same projection/package produces identical bytes.
8. Stage45 can reject a visually bad operator without corrupting/recomputing upstream mechanical or appearance artifacts.


## Ownership correction

See `AGENT_MEMORY/VISUAL_PRESENTATION_OWNERSHIP_CORRECTION.md`.

The recovery implementation must not accidentally make raw Stage18 source-mask CDT the final shipping render topology. The final visual presentation mesh/binding belongs after Stage35 qualification, naturally at Stage37, and is sealed by Stage38 before Stage42 runtime compilation.
