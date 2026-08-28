# RealSaS — Canonical IR Type System V1

**Date:** 2026-08-28  
**Status:** `CANONICAL_TYPED_AUTHORITY__EXECUTABLE_COMPILER_BOUNDARY`

## Core rule

> **Type encodes authority. Similar fields do not confer canonical authority.**

Authority classes are `EVIDENCE -> PROPOSAL -> QUALIFIED -> CANONICAL -> DERIVED -> RUNTIME`. No implicit `Proposal -> Canonical` conversion exists.

## Current executable chain

```text
ObservationEvidenceIR E
  ObservationSample(view, raster_xy, known ray origin/forward, learned depth d, support/provenance)
        ↓ deterministic analytic geometry + admitted persistence
RiggingSurfaceIR S
        ↓ Geppetto
SkeletonProposalIR G*
        ↓ Compiler canonical graph qualification
QualifiedSkeletonIR G
        ↓ Arachne
SkinProposalIR W*
        ↓ Compiler skin qualification
QualifiedSkinIR W
        ↓ Compiler assembly
CanonicalPuppetGraph Y
        ↓ exact-state proof / bounded repair / re-proof
ProofFrame + derived repair evidence
        ↓ only PASS proof bound to exact Y
RuntimePackageIR
```

## ObservationEvidenceIR

Owner: **IRIS + deterministic camera/raster bookkeeping**.

Current executable sample fields are:

```text
observation_id
view_index
raster_xy
ray_origin
ray_forward
depth                 # learned forward depth d
support
provenance_ref
validity_flags
```

`P` is not a learned field in the current executable boundary. SurfaceBuilder computes `P = O + dF` from known ray geometry. N-B3 closed the need for an explicit learned normal feature under the frozen tested regime; deterministic normals remain admissible only as qualified derived geometry when a consumer requires them.

Forbidden here: source-rig IDs, authored mechanical owner identity, model-issued product-canonical IDs.

## RiggingSurfaceIR

Owner: **deterministic SurfaceBuilder**.

Contains compiler-consumable surface nodes (`surface_id`, analytic `P`, support views, provenance, raster bindings, admitted persistence group, optional qualified derived normal), local relations and `geometry_lineage_hash`.

It is observation-grounded 2.5D geometry; it does not assert hidden watertight 3D truth.

## SkeletonProposalIR G*

Owner: **Geppetto**.

Proposal-local joint/edge IDs, positions, root/edge scores, confidence and surface support. `surface_binding_hash` must equal the exact consumed `RiggingSurfaceIR.geometry_lineage_hash`.

Proposal IDs are disposable and can never become product IDs by value equality.

## Compiler hierarchy qualification work layer

No duplicate skeleton-topology IR is introduced. The restored existing layer is:

```text
SkeletonProposalIR G*
  -> CanonicalGraphNodeCandidate / CanonicalGraphEdgeCandidate
  -> CanonicalGraphOptimizationRequest
  -> optimize_canonical_graph_v18_98
  -> CanonicalGraphOptimizationResult
  -> compiler-minted canonical joint IDs
  -> QualifiedSkeletonIR G
```

`realsas_topology.ShapeSkeletonGraph` is morphology/surface evidence and is **not** canonical product hierarchy.

## QualifiedSkeletonIR G

Owner: **Compiler rig authority**. Contains canonical compiler-owned joint IDs, positions, parent canonical IDs, one admitted root, support references, qualification report and `skeleton_lineage_hash`.

Minimum invariants: valid references, root policy, acyclicity/tree validity through the graph optimizer, recorded qualification, fail-closed on invalid proposal/solve.

## SkinProposalIR W*

Owner: **Arachne**. Binds to exact `surface_binding_hash` and exact `skeleton_binding_hash`; influences reference compiler-owned canonical joint IDs at the external qualification boundary.

## QualifiedSkinIR W

Owner: **Compiler skin authority**. Enforces finite/legal references and bounded simplex correction. Material negative weights, missing rows, unknown joints/surfaces or correction beyond the admitted budget fail closed.

The current Python skin qualifier is a safe executable qualification baseline, not a claim that it supersedes the stronger late-May BBW/QP/KKT numerical authority.

## CanonicalPuppetGraph Y

Owner: **Compiler Core only**; this is the single product truth.

Carries exact admitted S/G/W hashes, `product_state_hash`, optional `parent_state_hash`, qualification ledger, deformation/contact/motion state and editable metadata. Every accepted mutation creates a new product state and invalidates stale proof.

## ProofFrame / RepairDirective

Authority class: **DERIVED**. Proof binds to exact `product_state_hash`. A repair directive is not itself a mutation; a successful bounded repair creates `Y'`, which must be re-proven.

## RuntimePackageIR

Authority class: **RUNTIME PROJECTION**. Carries both `source_product_state_hash` and `source_proof_hash`. Export requires a PASS proof for the exact current product state. Runtime may optimize representation but can never redefine product truth.

## Lineage firewalls

Fail closed when proposal bindings are stale, solver/qualification invariants fail, proof refers to a different product state, or export lacks a PASS proof for the exact state.
