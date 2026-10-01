# Visual Presentation Ownership Correction

> NON-AUTHORITATIVE RECOVERY ARCHITECTURE DECISION CANDIDATE.

## Finding

The normalized architecture currently titles Stage18 as:

> Canonical mechanical mesh plus source-owned visual mesh, stable addressing and appearance domain

This is too strong for the latest Knight product-research evidence.

The final dynamic visual presentation topology cannot be fully determined at Stage18 because the late visual-deformation-region work requires **Stage35-qualified mechanical safe adjacency / boundary information**.

Stage35 itself depends on Stage18 mechanical mesh.

If final visual topology is made Stage18 authority and then later modified using Stage35 evidence, the architecture either:

- creates a hidden Stage35 → Stage18 back-edge / cycle; or
- pretends the later visual split is not a topology change; or
- ignores qualified safe-boundary evidence and permits visual triangles to cross mechanically unsafe regions.

The 2026-09-28 DAG truthfulness audit already demonstrated the general danger of Stage35→Stage18 side-channel repair cycles.

## Correct ownership split

### Stage18 — mechanical mesh + source visual substrate

Stage18 remains authority for:
- canonical mechanical mesh candidate;
- stable surface addressing;
- appearance domain;
- source-owned visual **substrate/candidate evidence** derived from source silhouette/raster.

Stage18 visual output is not yet the final dynamic presentation mesh.

It may continue to expose the current `VisualMeshSetIR.v1` during migration, but its semantic role must be treated as source-owned visual substrate/proposal, not shipping render topology.

### Stage35 — qualified mechanics

Stage35 remains authority for:
- frozen mechanical mesh dynamic qualification;
- exact safe mechanical adjacency / failure evidence;
- repair directives;
- no visual topology mutation.

### Stage37 — qualified visual presentation geometry + structure

Stage37 is the natural owner for final presentation geometry because it already consumes:
- qualified mechanics (Stage35);
- transferred skin (Stage36);
- skeleton (Stage28);
- appearance asset and qualification (Stage23/24);
- mechanical partition (Stage17).

Stage37 should additionally declare exact dependencies on the source visual evidence it consumes (Stage07 / Stage16 / Stage18 as needed).

It may produce:
- role-free presentation partition / slots / attachments;
- **QualifiedPresentationVisualMeshSetIR**;
- per-view safe deformation regions/charts;
- exact visual→mechanical binding;
- binding QA and source-topology provenance.

This keeps final render topology downstream of qualified mechanics without a cycle.

### Stage38 — atomic puppet seal

Stage38 seals:
- mechanical state;
- appearance;
- presentation structure;
- exact qualified visual presentation mesh/binding.

It records:
- `mechanical_mesh_render_authority = false` when source-owned visual presentation is active;
- exact qualified visual mesh hash;
- exact visual binding hash.

### Stage42 — projection / visual motion compilation

Stage42 does **not** invent or retriangulate visual topology.

It consumes the exact Stage38-qualified visual presentation geometry and:
- evaluates/compiles the selected visual-deformation operator over Stage41 motion;
- emits typed per-view/per-clip render geometry;
- packages exact source texture/CAA bindings;
- records operator implementation/policy identity.

Changing Stage42 deformation logic invalidates only Stage42–46.

### Stage45 — shipping visual quality authority

Stage45 decides whether the compiled visual motion is product-acceptable.

A bad visual operator is rejected here without changing Stage18 mechanical authority or re-fitting models.

## Why this is better

```text
Source / observation
       ↓
Stage18
  mechanical mesh
  visual substrate
       ↓
Stage35
  qualified mechanics
       ↓
Stage37
  QUALIFIED visual presentation topology/binding
       ↓
Stage38
  sealed complete puppet
       ↓
Stage42
  compile visual motion / runtime carrier
       ↓
Stage45
  dynamic visual qualification
```

This architecture:
- removes Stage35→Stage18 visual-repair pressure;
- matches the actual late Knight research dependency;
- makes visual experimentation cheap;
- preserves source-art ownership;
- keeps mechanical and visual authority distinct;
- allows presentation/runtime research to rerun without model fits or upstream geometry reconstruction.

## Existing evidence

The late trunk contains normalized generic primitives for:
- source-topology region construction;
- safe mechanical shared-edge reasoning;
- region-aware visual mesh construction;
- mechanical affine binding/evaluation;
- source-owned visual mesh IR;
- ARAP and alternative visual deformation QA.

The 1958 tail shows canonical mechanical support closes catastrophic mechanical stretch but substantial residual visual alpha/projection error remains. Therefore final visual topology/deformation must remain a presentation-stage qualification problem rather than being silently declared solved at Stage18.

## Recovery implication

Do not implement the Stage42 carrier by directly treating Stage18 `VisualMeshSetIR.v1` as final shipping geometry without qualification.

Recovery should first establish a Stage37-qualified visual presentation artifact, seal it at Stage38, then transport/compile it at Stage42.
