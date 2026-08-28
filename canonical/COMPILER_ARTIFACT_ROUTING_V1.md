# RealSaS — Compiler Artifact Routing V1

**Date:** 2026-08-28  
**Status:** `CANONICAL__TYPE_AUTHORITY_AND_PHYSICAL_BUNDLE_ROUTE`

An artifact is stored where its authority lives, and only its declared next stage may promote it. Similar fields do not imply interchangeable authority.

| Artifact | Producer | Bundle section | Consumer / promotion |
|---|---|---|---|
| `ObservationEvidenceIR` | IRIS + deterministic camera/raster bookkeeping | `ir/` | SurfaceBuilder only |
| `RiggingSurfaceIR` | deterministic SurfaceBuilder | `ir/` | Geppetto, Arachne, Compiler geometry consumers |
| `SkeletonProposalIR G*` | Geppetto | `candidates/` | Compiler rig/hierarchy qualification only |
| `CanonicalGraphOptimizationRequest/Result` | Compiler adapter + existing canonical graph optimizer | internal rig qualification evidence | materializes `QualifiedSkeletonIR`; never model truth |
| `QualifiedSkeletonIR G` | Compiler | `rig/` | Arachne + product assembly |
| `SkinProposalIR W*` | Arachne | `candidates/` | Compiler skin qualification only |
| `QualifiedSkinIR W` | Compiler | `weight/` | product assembly |
| `CanonicalPuppetGraph Y` | Compiler Core | `puppet/` | proof, bounded repair, export |
| `ProofFrame` / motion proof artifacts | proof system, bound to exact Y | `proof/` | repair/export only when current |
| `RepairDirective` | attribution/repair | `orchestrator/` | bounded Compiler repair only |
| `RuntimePackageIR` / `.rss/.rsr` | export projection of proven Y | `exports/` | C++ runtime only |

## Skeleton / topology clarification

No duplicate hierarchy layer is introduced. Historical `realsas_topology.shape_skeleton.ShapeSkeletonGraph` is morphology/surface evidence, **not** the canonical product joint hierarchy.

```text
Geppetto SkeletonProposalIR G*
        ↓ exact surface-lineage binding
CanonicalGraphNodeCandidate / CanonicalGraphEdgeCandidate
        ↓
CanonicalGraphOptimizationRequest
        ↓
optimize_canonical_graph_v18_98
  root selection + parent arborescence
  deterministic tie break
  optional bounded MILP shadow/escalation
        ↓
CanonicalGraphOptimizationResult
        ↓ compiler mints new product IDs
QualifiedSkeletonIR G
```

Proposal IDs and optimizer candidate IDs never become product-canonical IDs.

## Lineage firewalls

- Geppetto proposals carry exact `surface_binding_hash`.
- Arachne proposals carry exact `surface_binding_hash` and `skeleton_binding_hash`.
- Compiler rejects stale/mismatched proposals before solve.
- Skin rows reference compiler-owned canonical joint IDs only.
- Proof binds to exact `product_state_hash`.
- Runtime export requires a passing proof bound to that exact state and records both product and proof hashes.

## Physical implementation

`compiler/realsas_compiler_core/bundle_routes.py` reuses historical `realsas_artifacts.BundleWriter`; no second artifact store exists. Model proposals remain physically in `candidates/`, qualified/canonical artifacts in their authority sections.
