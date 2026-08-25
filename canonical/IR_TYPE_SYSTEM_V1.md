# RealSaS — Canonical IR Type System V1

**Date:** 2026-08-25  
**Status:** `CANONICAL_TYPED_BOUNDARY_SKELETON__FIELD_DETAILS_REVISIONABLE`

## 1. Purpose

RealSaS historically lost correctness when evidence, proposals, proof state and final product state could each act like separate truths. This type system prevents that class of regression.

Core rule:

> **Type encodes authority. A value cannot gain canonical authority merely because its fields look similar to a canonical value.**

## 2. Authority classes

```text
EVIDENCE     observation-grounded measurements / uncertainty / provenance
PROPOSAL     learned or heuristic candidate structure, not product truth
QUALIFIED    candidate that has passed explicit deterministic admission gates
CANONICAL    compiler-owned product state with canonical IDs and lineage
DERIVED      proof/diagnostic/attribution state bound to one canonical lineage
RUNTIME      serialization of one canonical lineage, never a second authority
```

Conversions must be explicit. There is no implicit `Proposal -> Canonical` cast.

## 3. ObservationEvidenceIR E

Owner: **IRIS + deterministic observation bookkeeping**.

Representative fields:

```text
ObservationEvidenceIR
  views[8]
  surface_samples[]
    P evidence
    N evidence where qualified
    support / visibility
    uncertainty / risk
    source view + pixel provenance
    persistence/cycle evidence
    optional retained hypotheses
  observation_frame
  evidence_version
```

Rules:

- no authored mechanical owner identity required;
- no source-rig IDs;
- no model-issued canonical product IDs;
- uncertainty/provenance must survive until intentionally consumed;
- camera metadata not consumed unless explicitly authorized by the active IRIS contract.

## 4. RiggingSurfaceIR S

Owner: **deterministic SurfaceBuilder / geometry qualification**.

Purpose: canonicalized observation-grounded geometry consumed by downstream rigging models.

Representative fields:

```text
RiggingSurfaceIR
  surface_nodes[]
    surface_id          # compiler/surface namespace, deterministic
    P
    N?                  # only when qualified
    support
    uncertainty
    provenance_refs[]
  local_relations[]
    neighbor
    same_sheet
    boundary/seam
    cross_view_same_locus
  sheets/components[]
  view_projection_bindings[]
  validity_flags[]
  geometry_lineage_hash
```

`RiggingSurfaceIR` does not imply a watertight hidden 3D mesh. It may be an oriented 2.5D surfel/sheet graph with deterministic view-local triangulation where useful.

## 5. SkeletonProposalIR G*

Owner: **Geppetto**.

```text
SkeletonProposalIR
  proposal_joints[]
    proposal_local_id
    position
    proposed_parent
    confidence
    supporting_surface_refs[]
  proposal_score
  model_provenance
```

Rules:

- proposal IDs are local and disposable;
- hierarchy may be invalid before qualification;
- source-rig exactness is not required;
- multiple valid candidates may exist.

## 6. QualifiedSkeletonIR G

Owner: **Compiler rig qualification**.

```text
QualifiedSkeletonIR
  joints[]
    canonical_joint_id
    position
    parent_canonical_id | ROOT
    surface_support_refs[]
    provenance
  root_id
  rig_invariants
  qualification_report_ref
  rig_lineage_hash
```

Hard minimum invariants:

- canonical IDs compiler-owned;
- valid parent references;
- acyclic hierarchy;
- root policy satisfied;
- duplicate/helper policy satisfied;
- bounded completion recorded explicitly;
- unsupported structure can fail closed.

## 7. SkinProposalIR W*

Owner: **Arachne**.

```text
SkinProposalIR
  surface_binding_ref
  skeleton_binding_ref
  proposed_influences[]
    surface_id
    [(proposal_joint_ref, weight), ...]
  confidence / uncertainty
  model_provenance
```

Rules:

- may violate exact normalization before qualification;
- may be dense/sparse/tokenized internally;
- must map externally to editable influences.

## 8. QualifiedSkinIR W

Owner: **Compiler skin qualification / solver-backed projection**.

```text
QualifiedSkinIR
  surface_binding_hash
  skeleton_binding_hash
  influences[]
    surface_id
    [(canonical_joint_id, weight), ...]
  normalization_residuals
  solver_report_ref?
  skin_lineage_hash
```

Hard minimum invariants:

- valid canonical joint references;
- finite weights;
- legal bounds/non-negativity policy;
- normalization/simplex policy;
- leakage/component policy;
- solver residual thresholds recorded;
- any deterministic correction is auditable.

## 9. CanonicalPuppetGraph Y

Owner: **Compiler Core only**.

This is the single product truth.

```text
CanonicalPuppetGraph
  product_lineage_id
  product_state_hash
  parent_state_hash?
  admitted_surface_ref
  admitted_skeleton_ref
  admitted_skin_ref
  deformation_state
  contact/physics_state
  motion_bindings
  editable_metadata
  qualification_ledger[]
  export_contract_version
```

Rules:

1. every product node has compiler-owned canonical identity;
2. every accepted mutation yields a new product state hash;
3. proof/repair/export bind to an exact product state hash;
4. no diagnostic object can mutate this graph without an admitted Compiler transition;
5. serialization is a projection of this graph, not a competing graph.

## 10. ProofFrame

Authority class: **DERIVED**.

```text
ProofFrame
  product_state_hash
  probe_plan_hash
  measurements
  proof_version
```

It is invalid if `product_state_hash` does not equal the product actually being qualified/exported.

## 11. MotionFailureSignatureSet

Authority class: **DERIVED**.

Contains typed failures such as deformation residual, foldover, contact violation, hierarchy/weight instability, playback invalidity or other verified signatures. It does not own product state.

## 12. OwnerAttributionReport

Authority class: **DERIVED**.

Attribution targets responsibility domains, not hidden teacher owners. Example domains:

```text
GEOMETRY
RIG_GRAPH
SKIN_WEIGHTS
DEFORMER
CONTACT
MOTION_BINDING
EXPORT_RUNTIME
UNKNOWN / ABSTAIN
```

It may identify canonical node references from the bound product lineage.

## 13. RepairDirective

Authority class: **DERIVED COMMAND**, not a mutation itself.

```text
RepairDirective
  source_product_state_hash
  owner_domain
  target_refs[]
  allowed_operation_class
  bounded_budget
  expected_failure_signature
  abstain_if_preconditions_fail
```

Compiler validates preconditions before applying it. Successful application creates **new canonical state `Y'`** and requires re-proof.

## 14. RuntimePackageIR

Authority class: **RUNTIME PROJECTION**.

```text
RuntimePackageIR
  source_product_state_hash
  manifest
  textures/assets
  rig/skin/deformation payload
  .rsr payload
  ABI/schema version
```

Runtime package must carry or cryptographically bind the source product state identity. Runtime adapters may optimize representation but cannot redefine product truth.

## 15. ID namespaces

At minimum:

```text
OBSERVATION_LOCAL     disposable observation/sample IDs
PROPOSAL_LOCAL        disposable Geppetto/Arachne IDs
SURFACE_CANONICAL     deterministic SurfaceBuilder/Compiler namespace
PRODUCT_CANONICAL     Compiler-owned joint/control/component/product IDs
PROOF_LOCAL           derived probe/failure IDs
RUNTIME_LOCAL         serialization handles bound back to product IDs
```

Forbidden: treating a proposal-local ID as product-canonical because values happen to align.

## 16. Lineage rules

Every canonical transition records:

```text
parent_state_hash
operation / solver / compiler stage
input qualified IR hashes
configuration hash
output product_state_hash
residual/invariant report hashes
```

This ledger is the backbone for reproducible repair and proof.

## 17. Fail-closed transitions

A transition does not produce a qualified/canonical output when:

- required invariants fail;
- solver residuals exceed admitted thresholds;
- referenced IR hashes do not match;
- uncertainty/support is insufficient for the requested operation;
- repair preconditions refer to a stale lineage;
- a runtime export cannot bind to the proven final state.

In these cases the correct output is a typed failure/abstention record, not a partially canonical object.
