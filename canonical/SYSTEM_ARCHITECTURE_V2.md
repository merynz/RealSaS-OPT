# RealSaS — Canonical System Architecture V2

**Updated:** 2026-08-31  
**Status:** `CANONICAL_RESPONSIBILITY_AND_IR_ARCHITECTURE__DOWNSTREAM_OWNERSHIP_AUDITED__MESH_WEIGHT_TYPED_SEAM_OPEN`

`PRODUCT_CONTRACT_V1.md` owns product responsibility. `IR_TYPE_SYSTEM_V1.md` owns current typed authority. `DETERMINISTIC_DOWNSTREAM_LAYER_OWNERSHIP_OVERLAP_AUDIT_20260831.md` owns the downstream non-duplication findings.

## Canonical system

```text
8 ordered neutral views + known orthographic cameras
        ↓
IRIS — learned observation-grounded geometry evidence
        ↓ analytic P = O + dF
GeometricSubstrateAssembler
(historical implementation name: SurfaceBuilder)
        ↓ RiggingSurfaceIR S
Geppetto
        ↓ SkeletonProposalIR G*
Compiler graph qualification / global optimizer
        ↓ QualifiedSkeletonIR G
Arachne
        ↓ SkinProposalIR W*
Compiler bounded skin qualification
        ↓ QualifiedSkinIR W
[typed editable-mesh + mesh-weight binding seam — REQUIRED, exact schema unsealed]
        ↓
Compiler Core
        ↓ CanonicalPuppetGraph Y
ARAP / contact execution + motion/deformation proof
        ↓ failure attribution
owner-routed bounded repair -> new Y' -> mandatory re-proof
        ↓ only proven exact Y
engine-neutral runtime projection
```

Responsibility boundaries do not imply permanently separate neural checkpoints or processes.

## Authority invariants

- IRIS/Geppetto/Arachne produce evidence or proposals, never final product truth.
- Deterministic geometry assembly may derive geometry but not mechanical hierarchy/skin semantics.
- Compiler graph qualification globally selects/canonicalizes Geppetto evidence; it does not become a second generic skeleton estimator.
- Compiler skin qualification legally/mathematically projects Arachne evidence within bounded policy; it does not become a hidden second semantic weight estimator.
- A full deterministic BBW/QP/KKT synthesis, if deliberately used, is a separately typed proposal/fallback arm rather than invisible qualification.
- CDT is a derived discretization operator, not hidden-surface truth.
- Compiler owns canonical product IDs and the single `CanonicalPuppetGraph` lineage.
- ARAP/XPBD/contact/proof are exact-state downstream operators; they may not silently rewrite canonical rest truth.
- Proof, repair, serialization, export and runtime projection bind to the exact same product state.
- Diagnostics and proof artifacts are derived/non-owning.
- Runtime is a projection, not a second canonical graph.
- Stale proposal/proof bindings fail closed.

## IR layers

Current executable authority classes remain:

`EVIDENCE -> PROPOSAL -> QUALIFIED -> CANONICAL -> DERIVED -> RUNTIME`.

Current hierarchy qualification deliberately reuses the restored historical graph optimizer. There is no duplicate topology/hierarchy authority:

```text
G* -> CanonicalGraphOptimizationRequest/Result -> QualifiedSkeletonIR G
```

`ShapeSkeletonGraph` and `RiggingSurfaceIR.local_relations` remain morphology/local geometric evidence only.

### Known IR V1 gap

The executable V1 type library has no explicit editable mesh/discretization type and no explicit qualified-skin-to-mesh-weight binding type. This is now a named contract gap rather than an implicit historical path.

Before final Arachne/product seal, `MESH_WEIGHT_BINDING_CONTRACT` must define:

- mesh/discretization ownership and lineage to `RiggingSurfaceIR`;
- CDT/topology policy and UNKNOWN behavior;
- qualified skin field/row query or transfer to mesh elements;
- coverage/residual/failure semantics;
- BBW/QP/KKT normal-path versus fallback role;
- how mesh/weight state is included in the single product hash and invalidates proof.

Exact type names are intentionally unsealed until that contract is written.

## GeometricSubstrateAssembler boundary

Current learned geometric authority remains observation evidence/depth under IRIS's active V2 contract; common-frame point geometry is analytic from known rays. Support/provenance and admitted persistence remain deterministic/observation-grounded.

The assembler may expose additional deterministic local geometry when a demonstrated consumer need justifies it, but may not silently absorb Geppetto/Arachne mechanical responsibilities or hidden completion.

## Geppetto boundary

`RiggingSurfaceIR S -> SkeletonProposalIR G*`.

Geppetto proposes control geometry/count/existence and root/directed-edge evidence. Compiler validates binding, solves the admitted global root/parent graph and mints product-canonical IDs only after qualification.

## Arachne boundary

`RiggingSurfaceIR S + QualifiedSkeletonIR G -> SkinProposalIR W*`.

Arachne proposes the semantic influence field/weights. Compiler owns legality, bounded mathematical projection, residual reporting and fail-closed behavior.

A numerical solver that materially creates a new semantic skin field is not qualification; it must be an explicit proposal producer/fallback arm.

## Mesh / weight projection boundary

A mesh is an editable/deformation/render discretization of admitted geometry, not a second observed geometry truth. A mesh-weight binding is a deterministic query/interpolation/projection of admitted qualified skin semantics, not a second Arachne.

Until the explicit typed seam is frozen, no historical CDT/BBW wiring may silently become final product authority.

## Compiler Core / deformation / proof / repair

Compiler is the only canonical product owner after proposal stages. The proof loop is:

```text
Y -> ResolvedMotionProbePlan -> measurements -> proof
PASS -> export allowed
FAIL -> failure signatures -> attribution -> owner-scoped RepairDirective -> Y' or abstain -> re-proof
```

ARAP/XPBD/contact may produce posed/corrected derived state and residuals from exact Y. Any accepted rest geometry/skeleton/skin/mesh mutation creates new Y'. No repair against stale Y may authorize export of another state.

Repair is an orchestrated owner-routed process, not a parallel monolithic auto-rigger.

## Runtime boundary

The restored v0.5 C++17 SDK remains the engine-neutral historical runtime baseline. Host engines do not own canonical product truth.

## Historical composition and current execution distinction

Restoration is a controlled composition, not “newest wins”:

- Aug-9 v0.5: executable compiler/proof/export/runtime chassis;
- Aug-6 R5_3: compiler-owned identity/reducer/DAG authority discipline;
- May v97.39: single product/proof/repair/export truth invariant;
- late-May v97.43: stronger numerical reference for CDT/cotangent, BBW/QP/KKT, ARAP and XPBD/contact where parity with later ports is unproven.

The current GitHub facade self-contains typed surface/rig/skin/product code and a narrow graph/contracts vendor closure. It does **not** contain the historical `realsas_mesh`, `realsas_weight` or `realsas_deformation` packages named by `solver_registry.py`, and that registry is not imported by the canonical package entrypoint.

Therefore historical mesh/weight/deformation solvers remain restoration/SHA authorities until an exact typed executable integration is restored and promoted. “Historical executable baseline” must not be confused with “present in a clean current-main checkout.”

## Restoration / continuation status

Graph and typed proposal/product routing are in the current canonical loop. Historical numerical regressions/provenance remain valuable, but final modern mesh/weight integration now has an explicit contract gate.

Open work is therefore not a generic Compiler restoration. It is:

1. active IRIS V2 gates;
2. Geppetto/Arachne R6 oracle ceilings under their frozen reference audit;
3. close the mesh/mesh-weight typed seam before final Arachne/product architecture seal;
4. promote any historical stronger numerical operator only with typed authority, residual/invariant and downstream parity evidence.
