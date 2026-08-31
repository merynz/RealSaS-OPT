# RealSaS — MWB-0 Typed Mesh / Weight Seam Implementation Closure — 2026-09-01

**Status:** `MWB0_CLOSED_PASS__TYPED_SCHEMA_HASHING_ARTIFACT_ROUTE_PREFLIGHT_IMPLEMENTED__MWB1_NEXT`

## Purpose

Close implementation gate MWB-0 from `canonical/MESH_WEIGHT_BINDING_CONTRACT_V1.md` without promoting any historical numerical solver or changing learned-system authority.

## Implemented executable surface

The current Compiler now exposes the following typed seam:

- `SurfaceSupportBinding`;
- `MeshDiscretizationCandidateIR`;
- `QualifiedEditableMeshIR`;
- `QualifiedMeshSkinIR`;
- `CanonicalPuppetGraphV2` with schema `RealSaS.CanonicalPuppetGraph.v2`.

The legacy `CanonicalPuppetGraph.v1` assembly path remains available during migration so the previously qualified sacrificial consumer route is not silently rewritten.

## Binding invariants implemented

The MWB-0 executable layer now enforces:

- mesh-to-surface lineage equality;
- mesh-skin binding to exact surface, skeleton, qualified-skin and qualified-mesh hashes;
- non-empty finite nonnegative support coefficients with simplex sum 1;
- exact identity-binding semantics for `IDENTITY_SURFACE_NODE`;
- rest mesh positions derived only from admitted `RiggingSurfaceIR` support;
- topology reference validity for vertices/faces/edges;
- complete one-row-per-qualified-mesh-vertex mesh-skin coverage;
- legal canonical-joint references;
- finite/nonnegative/simplex mesh-weight rows;
- deterministic candidate/mesh/mesh-skin lineage hashing;
- product-state identity binding admitted surface, skeleton, skin, mesh and mesh-skin hashes;
- stale proof rejection after product-state mutation.

These validators are admission/invariant code. They do not generate topology or synthesize skin semantics.

## Artifact routing implemented

Physical routing now distinguishes:

- mesh candidate: `candidates/mesh/mesh_discretization_candidate_ir.json`;
- qualified editable mesh: `mesh/qualified_editable_mesh_ir.json`;
- qualified mesh skin: `weight/qualified_mesh_skin_ir.json`;
- V2 canonical product: `puppet/canonical_puppet_graph_v2.json`.

Runtime remains a projection of a passing proof bound to the exact current product-state hash.

## Explicitly NOT promoted by MWB-0

MWB-0 does not implement or authorize:

- CDT/triangulation as a product generator;
- hidden geometry completion;
- BBW/QP/KKT semantic skin generation;
- ARAP/XPBD numerical promotion;
- Arachne predictor behavior;
- any learned optimizer step.

Historical numerical authorities remain separately provenance-bound until their own promotion gates.

## Regression evidence

Pull request: `#11 Implement MWB-0 typed mesh and weight seam`.

Merged squash commit: `c87502a7d106681a41802ca1adf3aafa2f669a55`.

PR head CI:

- `mwb0-typed-seam-v1` run `33439698661`: **PASS**;
- `consumer-interlock-v0` run `33439698705`: **PASS**;
- `consumer-coupling-probe-v1` run `33439698523`: **PASS**.

The new MWB-0 regressions cover:

1. legacy product route remains V1-compatible;
2. V2 product binds mesh + mesh-skin hashes;
3. topology mutation changes mesh/product state and rejects stale proof;
4. mesh-skin mesh-lineage mismatch fails closed;
5. missing mesh-vertex weight rows fail closed;
6. invalid support-binding simplex fails closed.

## Scientific/product interpretation

This closure proves the missing typed seam can be represented, hashed, routed and invalidated coherently in the current Compiler facade. It does **not** prove that a useful editable mesh can yet be generated from admitted surface evidence, nor that Arachne field-to-mesh transfer preserves deformation quality.

Therefore the next gate is MWB-1, not product seal.

## Next gate

`MWB-1_IDENTITY_SUBSET_DETERMINISTIC_BASELINE`

Required real-witness proof:

- construct a mesh only from identity-bound admitted surface nodes;
- no new geometry authority;
- topology validity;
- exact surface lineage;
- exact qualified-skin row copy to mesh vertices;
- product hash mutation under topology/weight mutation;
- stale proof rejection on the real route.

Only after MWB-1 passes may MWB-2 open local convex inserted vertices and/or CDT promotion.
