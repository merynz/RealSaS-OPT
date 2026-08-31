# RealSaS — Canonical IR Type System V1

**Updated:** 2026-08-31  
**Status:** `CANONICAL_TYPED_AUTHORITY__EXECUTABLE_COMPILER_BOUNDARY__MESH_WEIGHT_SEAM_EXPLICITLY_OPEN`

## Core rule

> **Type encodes authority. Similar fields do not confer canonical authority.**

Authority classes are `EVIDENCE -> PROPOSAL -> QUALIFIED -> CANONICAL -> DERIVED -> RUNTIME`. No implicit `Proposal -> Canonical` conversion exists.

## Current executable typed chain

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
        ↓ [OPEN TYPED SEAM: editable mesh/discretization + qualified skin-to-mesh binding]
        ↓ Compiler assembly
CanonicalPuppetGraph Y
        ↓ exact-state proof / bounded owner-routed repair / re-proof
ProofFrame + derived repair evidence
        ↓ only PASS proof bound to exact Y
RuntimePackageIR
```

The bracketed mesh/weight seam is an explicit known gap in IR V1, not permission for an implicit historical solver path. Exact type names/schemas are intentionally unsealed until `MESH_WEIGHT_BINDING_CONTRACT` is frozen.

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

`P` is not a learned field in the current executable boundary. The historical implementation name `SurfaceBuilder` computes `P = O + dF` from known ray geometry; planned functional nomenclature is `GeometricSubstrateAssembler`. Deterministic normals remain admissible only as qualified derived geometry when a consumer requires them.

Forbidden here: source-rig IDs, authored mechanical owner identity, model-issued product-canonical IDs.

## RiggingSurfaceIR

Owner: **deterministic GeometricSubstrateAssembler** (historical implementation name `SurfaceBuilder`).

Contains compiler-consumable surface nodes (`surface_id`, analytic `P`, support views, provenance, raster bindings, admitted persistence group, optional qualified derived normal), local relations and `geometry_lineage_hash`.

It is observation-grounded partial geometry; it does not assert hidden watertight 3D truth.

`SurfaceRelation` is geometric/morphological evidence only. A `relation_kind` may not confer root/parent/bone/mechanical hierarchy authority.

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

Arachne owns semantic influence proposal. A deterministic full BBW-generated skin, if deliberately used, is a separately typed proposal arm and may not be smuggled into qualification as a second semantic owner.

## QualifiedSkinIR W

Owner: **Compiler skin authority**. Enforces finite/legal references and bounded mathematical/simplex/influence projection. Material negative weights, missing rows, unknown joints/surfaces or correction beyond admitted budget fail closed.

The current Python qualifier is the self-contained executable qualification baseline. Stronger historical BBW/QP/KKT numerical authority remains a reference/promotion target, but when used in the normal Arachne path it must preserve Arachne semantic evidence within a frozen bounded projection budget. If it materially synthesizes new influence ownership, its output is proposal evidence, not automatically `QualifiedSkinIR`.

## Open typed mesh/discretization seam

The current `types.py` intentionally has **no explicit mesh IR**. Historical compiler work includes view-local triangulation/CDT, but final product integration must not remain implicit.

Before final Arachne/product seal, freeze an explicit typed contract for:

- editable/view-local mesh or deformation discretization;
- its relation to `RiggingSurfaceIR` and UNKNOWN/unobserved regions;
- topology lineage/provenance;
- qualified skin-field/row evaluation or transfer to mesh vertices/elements;
- transfer coverage/residual/failure behavior;
- whether any numerical weight solver is bounded projection or a separately typed proposal producer;
- exact inclusion in `CanonicalPuppetGraph` state hashing and proof invalidation.

A triangulated mesh is not a second observed-geometry truth. Mesh-weight transfer is not a second Arachne.

## CanonicalPuppetGraph Y

Owner: **Compiler Core only**; this is the single product truth.

Current V1 object carries exact admitted S/G/W hashes, `product_state_hash`, optional `parent_state_hash`, qualification ledger, deformation/contact/motion state and editable metadata. Every accepted mutation creates a new product state and invalidates stale proof.

The absence of explicit mesh fields in V1 is a known typed-seam gap, not evidence that mesh may be maintained as a parallel canonical truth. The future mesh/weight binding contract must feed the single Y lineage.

## ProofFrame / RepairDirective

Authority class: **DERIVED**. Proof binds to exact `product_state_hash`. A repair directive is not itself a mutation; a successful bounded owner-specific repair creates `Y'`, which must be re-proven.

Repair may not become a hidden generic auto-rigger that independently changes geometry, hierarchy and weights until proof passes.

## RuntimePackageIR

Authority class: **RUNTIME PROJECTION**. Carries both `source_product_state_hash` and `source_proof_hash`. Export requires a PASS proof for the exact current product state. Runtime may optimize representation but can never redefine product truth.

## Lineage firewalls

Fail closed when proposal bindings are stale, solver/qualification invariants fail, mesh/weight projection lacks the future required lineage/residual contract, proof refers to a different product state, or export lacks a PASS proof for the exact state.

Downstream overlap authority: `DETERMINISTIC_DOWNSTREAM_LAYER_OWNERSHIP_OVERLAP_AUDIT_20260831.md`.
