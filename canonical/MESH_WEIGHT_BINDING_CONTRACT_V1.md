# RealSaS — Mesh / Weight Binding Contract V1

**Date:** 2026-08-31  
**Status:** `FROZEN_PREIMPLEMENTATION__NORMAL_PATH_OWNERSHIP_CLOSED__TYPED_SEAM_REQUIRED_BEFORE_FINAL_ARACHNE_PRODUCT_SEAL`

## Purpose

Close the missing typed seam between observation-grounded surface evidence, editable product mesh topology, Arachne skin fields/rows, mesh-vertex weights, deformation solvers and the single canonical product state.

This contract is a consequence of the downstream ownership-overlap audit. It is intentionally frozen before implementing or promoting a mesh/weight solver path.

## External and historical evidence credited

- RigAnything shows that skeleton inference itself can consume complete-surface `P+N` without polygon connectivity, while its released final skin path additionally uses deterministic mesh-neighbor smoothing and transfer.
- SkinTokens is the cleaner Arachne representation reference: learned geometry-conditioned per-bone fields can be queried at requested surface points; sampled-to-original-mesh nearest-neighbor transfer is a deterministic helper, not the learned skin function.
- Current RealSaS `QualifiedSkinIR` is indexed by `surface_id` and Compiler qualification owns lineage/legal/simplex policy.
- Historical RealSaS retains CDT, BBW/QP/KKT, ARAP and XPBD as numerical authorities/references, but the current typed IR library does not yet expose an explicit mesh or mesh-weight binding artifact.

No external implementation is copied by this contract.

---

# 1. Binding authority decomposition

## 1.1 Geometric truth

`RiggingSurfaceIR S` remains the only admitted observation-grounded rest-geometry substrate below IRIS.

Mesh construction may discretize `S`; it may not create a second hidden-surface truth.

## 1.2 Mesh topology

Editable mesh topology is a **product component** because it affects editability, deformation and rendering. It is nevertheless **not geometry authority**: every mesh vertex and face must carry exact lineage back to admitted `S` support.

Therefore the final product path requires a qualified mesh artifact whose topology hash participates in product-state identity.

## 1.3 Skin semantics

Arachne owns learned semantic influence evidence conditioned on exact `S + QualifiedSkeletonIR G`.

Compiler skin qualification owns admissibility/legal references/simplex/sparsity policy. It may not independently synthesize a materially different semantic skin solution in the normal learned path.

## 1.4 Mesh-vertex weight binding

Mapping qualified skin evidence from the admitted surface domain to the editable mesh is deterministic and separately typed. The binder may interpolate already-qualified weights; it may not become a second skin predictor.

## 1.5 Deformation / contact

ARAP, XPBD/contact and motion proof consume the exact admitted mesh + skeleton + mesh-weight state. They may report residual/failure evidence but may not silently rewrite rest topology, skeleton or skin semantics.

---

# 2. Required typed artifacts

Names are frozen for V1 unless an explicit schema revision is made.

## 2.1 `MeshDiscretizationCandidateIR M*`

Authority class: `DERIVED_CANDIDATE`.

Producer: deterministic mesh/discretization operator from exact `RiggingSurfaceIR + view + frozen mesh policy`.

Minimum fields:

- candidate lineage/config hash;
- exact `surface_binding_hash`;
- view index / camera binding;
- candidate vertices;
- candidate faces/edges;
- per-vertex `SurfaceSupportBinding`;
- boundary/constraint provenance;
- coverage/UNKNOWN diagnostics;
- solver provenance and residual/report envelope.

This type does not claim canonical product truth.

## 2.2 `SurfaceSupportBinding`

For every mesh vertex `v`, define nonnegative coefficients:

`a(v,s) >= 0`, `sum_s a(v,s) = 1`

over admitted `surface_id` values.

Rest position is derived only from admitted support:

`P_v = sum_s a(v,s) P_s`.

Two admissible cases exist:

1. `IDENTITY_SURFACE_NODE`: one coefficient is exactly 1;
2. `LOCAL_CONVEX_INTERPOLATION`: multiple coefficients form a frozen local convex interpolation over mutually compatible admitted support.

No unconstrained nearest-neighbour jump across disconnected components, no support through `UNKNOWN`, and no hidden completed surface is admitted.

## 2.3 `QualifiedEditableMeshIR M`

Authority class: `QUALIFIED_PRODUCT_COMPONENT`.

Owner: Compiler geometry qualification.

Contains:

- canonical mesh/vertex/face IDs minted after qualification;
- exact surface lineage;
- exact view/camera binding;
- rest positions derived from `SurfaceSupportBinding`;
- topology and boundary constraints;
- support/coverage classification;
- qualification/solver report;
- `mesh_lineage_hash`.

A topology mutation changes `mesh_lineage_hash` even if all rest positions are unchanged.

## 2.4 `QualifiedMeshSkinIR B`

Authority class: `QUALIFIED_PRODUCT_COMPONENT`.

Producer/owner: deterministic mesh-weight binder + Compiler qualification.

Bindings:

- exact `surface_binding_hash`;
- exact `skeleton_binding_hash`;
- exact `skin_binding_hash` (`QualifiedSkinIR.skin_lineage_hash`);
- exact `mesh_binding_hash` (`QualifiedEditableMeshIR.mesh_lineage_hash`).

Rows are indexed by canonical mesh-vertex ID and reference Compiler-owned canonical joint IDs.

The artifact records:

- transfer method;
- source support coefficients;
- finite/nonnegative/simplex residuals;
- any bounded numerical correction;
- unsupported/UNKNOWN failures;
- `mesh_skin_lineage_hash`.

---

# 3. Normal product path

The frozen normal path is:

```text
RiggingSurfaceIR S
   |\
   | \-> deterministic mesh discretizer -> MeshDiscretizationCandidateIR M*
   |                                      -> Compiler geometry qualification
   |                                      -> QualifiedEditableMeshIR M
   |
   -> Geppetto -> SkeletonProposalIR G*
                -> Compiler graph qualification
                -> QualifiedSkeletonIR G

S + G -> Arachne -> SkinProposalIR W*
                 -> Compiler skin qualification
                 -> QualifiedSkinIR W

M + W -> deterministic mesh-weight binder
      -> QualifiedMeshSkinIR B

S + G + W + M + B
      -> CanonicalPuppetGraph V2 Y
      -> ARAP / XPBD / motion proof
      -> PASS-only runtime projection
```

Mesh generation can execute in parallel with Geppetto/Arachne after `S`, but canonical product assembly waits for all admitted component hashes.

---

# 4. Deterministic mesh rule

## 4.1 Initial topology class

The initial implementation target is view-local constrained triangulation over admitted support.

CDT is a numerical method candidate/reference, not semantic authority.

## 4.2 Forbidden completion

A triangle must not become evidence for a surface region merely because a triangulator can connect its vertices.

Faces crossing background, unsupported holes, disconnected components or typed `UNKNOWN` are forbidden unless a future explicit completion/product policy is separately authorized.

## 4.3 Steiner / inserted vertices

Inserted vertices are allowed only when they receive an admissible `SurfaceSupportBinding` and their rest position is deterministically derived from admitted support.

If such binding cannot be established, the mesh operation fails closed or leaves the region undiscretized.

## 4.4 Historical CDT promotion

The late-May exact-predicate CDT/cotangent implementation remains the numerical reference. It becomes current product code only after exact source restoration plus typed-lineage, topology-invariant and downstream deformation parity evidence.

---

# 5. Skin-to-mesh transfer rule

For a mesh vertex with support coefficients `a(v,s)` and qualified surface skin rows `w(s,j)`, the default deterministic transfer is:

`w_raw(v,j) = sum_s a(v,s) w(s,j)`.

Because both `a(v,.)` and qualified skin rows are simplex-valued, the exact mathematical result is also simplex-valued. Numerical residuals must be measured rather than silently normalized away.

### Identity case

For `IDENTITY_SURFACE_NODE`, mesh weights must be bitwise/declared-numerically equivalent to the source qualified skin row modulo serialization precision.

### Convex interpolation case

For `LOCAL_CONVEX_INTERPOLATION`, the binder may only combine the explicitly bound source rows. It may not introduce a new joint support absent from every contributing source row.

### Unsupported case

If a mesh vertex lacks valid admitted support, the normal path must fail/abstain with typed attribution. It must not invoke unconstrained BBW merely to fill the row.

---

# 6. BBW / QP / KKT role

## 6.1 Normal Arachne path

Full semantic BBW generation is **not authorized** inside `Compiler.skin_qualification` or the mesh-weight binder.

Optional numerical projection/regularization may be considered later only if it is proven to preserve Arachne semantic support within a preregistered bounded correction budget and improves a demonstrated failure.

No such mechanism is authorized by V1 merely because historical code exists.

## 6.2 Full deterministic BBW alternative

If a future experiment uses BBW as a full skin generator from geometry/handles, it is an explicit alternative proposal/fallback arm, not hidden qualification.

That arm requires its own typed proposal contract, provenance, causal comparison against Arachne, Compiler qualification and deformation proof before it can enter a product path.

## 6.3 Historical numerical authority

Late-May BBW/QP/KKT remains a strong numerical reference/upper-bound source until exact current typed restoration/parity is demonstrated. Historical strength does not grant silent semantic ownership.

---

# 7. Canonical product state revision

Current `CanonicalPuppetGraph.v1` binds only admitted surface/skeleton/skin hashes explicitly. Final mesh-based product assembly therefore requires a V2 schema before shipping/product seal.

`CanonicalPuppetGraph.v2` must explicitly bind at minimum:

- `admitted_surface_hash`;
- `admitted_skeleton_hash`;
- `admitted_skin_hash`;
- `admitted_mesh_hash`;
- `admitted_mesh_skin_hash`;
- qualification ledger for mesh and mesh-skin admission;
- parent state hash;
- exact product state hash.

Mesh or mesh-weight authority must **not** be hidden inside generic `deformation_state`, `editable_metadata` or runtime payload fields.

Every admitted rest mesh/weight mutation creates a new product state and invalidates stale proof.

---

# 8. Artifact routing revision

Before implementation, physical bundle routing must gain:

- `mesh/mesh_discretization_candidate_ir.json` or candidate-equivalent under `candidates/mesh/`;
- `mesh/qualified_editable_mesh_ir.json`;
- `weight/qualified_mesh_skin_ir.json`.

Runtime consumes only PASS-proven product state. It may repack/optimize these artifacts but may not redefine them.

---

# 9. Fail-closed attribution classes

At minimum:

- `MESH_SURFACE_LINEAGE_MISMATCH`;
- `MESH_VERTEX_SUPPORT_INVALID`;
- `MESH_FACE_CROSSES_UNSUPPORTED_DOMAIN`;
- `MESH_TOPOLOGY_INVALID`;
- `MESH_WEIGHT_SKIN_LINEAGE_MISMATCH`;
- `MESH_WEIGHT_MESH_LINEAGE_MISMATCH`;
- `MESH_WEIGHT_UNSUPPORTED_VERTEX`;
- `MESH_WEIGHT_ILLEGAL_JOINT_REFERENCE`;
- `MESH_WEIGHT_SIMPLEX_RESIDUAL`;
- `MESH_WEIGHT_TRANSFER_CORRECTION_BUDGET_EXCEEDED`.

Failure attribution routes back to the semantic owner. Mesh failure does not authorize arbitrary skin repair; skin failure does not authorize hidden geometry completion.

---

# 10. Required implementation gates

## MWB-0 — typed schema / hashing / artifact-route preflight

Implement the four binding concepts (`M*`, support binding, `M`, `B`) plus `CanonicalPuppetGraph.v2` and exact stale-lineage tests. No numerical solver promotion is required.

## MWB-1 — identity/subset deterministic baseline

On a real witness, build a mesh using only identity-bound admitted surface nodes. Prove:

- no new geometry authority;
- topology validity;
- exact surface lineage;
- exact weight-copy path;
- product hash changes under topology/weight mutation;
- stale proof rejection.

## MWB-2 — constrained interpolation / CDT candidate

Only after MWB-1, add local convex inserted vertices and/or promoted CDT. Measure unsupported-domain crossing, topology validity and deformation non-inferiority.

## MWB-3 — Arachne field-to-mesh transfer ceiling

Before Arachne predictor training is considered product-ready, prove that authoritative/codec skin evidence can be decoded on admitted `S`, transferred to `M`, and retain deformation quality below the frozen consumer thresholds.

## MWB-4 — historical solver promotion, only if needed

Promote exact-predicate CDT, BBW/QP/KKT, ARAP or XPBD components one solver at a time through source provenance + typed compatibility + invariant/residual + downstream parity gates.

No solver is promoted solely because it is historically more sophisticated.

---

# 11. Training and scope

This contract:

- does not authorize learned optimizer steps;
- does not change IRIS Gate0;
- does not change Geppetto R6 causal arms;
- does not require BBW in the normal Arachne path;
- does not require complete hidden mesh surface;
- blocks final Arachne/product architecture seal until MWB-0 is implemented and regression-tested.

Program rule remains binding:

> `NO_ARCHITECTURAL_MECHANISM_WITHOUT_A_DEMONSTRATED_FAILURE_IT_ADDRESSES`.
