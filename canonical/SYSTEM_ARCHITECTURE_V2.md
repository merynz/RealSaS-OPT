# RealSaS — Canonical System Architecture V2

**Date:** 2026-08-25  
**Status:** `CANONICAL_RESPONSIBILITY_AND_IR_SKELETON__EXECUTION_GATE_TOPOLOGY_EVIDENCE_CONTROLLED`  
**Scope:** system responsibilities, typed IR boundaries, canonical authority, solver placement, proof/repair lineage, runtime boundary

This document refines the internal system architecture without changing the shipping objective or returning hidden mechanical ontology to IRIS. `PRODUCT_CONTRACT_V1.md` remains the responsibility-level product contract and `OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md` remains the IRIS problem-definition authority.

## 1. North star

```text
ONE neutral 8-view character sheet
        ↓
observable 2.5D geometry
        ↓
clean editable rig proposal
        ↓
clean editable skin proposal
        ↓
deterministic qualification / proof / repair
        ↓
verified editable 2D puppet + engine-neutral runtime package
```

Coordinate dimension may be 3D while the object model remains observation-grounded 2.5D and the shipping puppet remains 2D/8-view.

## 2. Canonical responsibility architecture

```text
8 ordered neutral-pose RGBA views X
        │
        ▼
      IRIS
        │  ObservationEvidenceIR E
        ▼
 deterministic SurfaceBuilder
        │  RiggingSurfaceIR S
        ▼
     Geppetto
        │  SkeletonProposalIR G*
        ▼
 skeleton qualification / normalization
        │  QualifiedSkeletonIR G
        ▼
      Arachne
        │  SkinProposalIR W*
        ▼
 skin qualification / normalization
        │  QualifiedSkinIR W
        ▼
   Compiler Core
        │  CanonicalPuppetGraph Y
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
   Motion Proof      Repair          Export
        │              │               │
        └──── exact same Y lineage ─────┘
                       │
                       ▼
              verified editable puppet
                       │
                       ▼
           .rss/.realsas + .rsr package
                       │
                       ▼
                 C++ runtime ABI
```

The boxes above are **responsibility boundaries**, not a permanent requirement for separate processes or checkpoints.

## 3. Proposal is not truth

Learned stages produce proposals/evidence. They do not create canonical product truth.

```text
IRIS evidence            = observation-grounded evidence
Geppetto output G*       = skeleton proposal
Arachne output W*        = skin proposal
Compiler-qualified S/G/W = admitted typed state
CanonicalPuppetGraph Y   = single product truth
```

A proposal may be excellent and still fail qualification. A rejected proposal must never be serialized under a production-looking canonical identity.

## 4. Single-truth invariant

The restored compiler inherits the strongest historical authority rule:

> Proof, repair, serialization, export and runtime packaging must all refer to the exact same canonical product lineage.

Forbidden:

- proof graph A while final product graph is B;
- parallel canonical owners for the same stage;
- model-issued canonical product IDs bypassing compiler ownership;
- diagnostics mutating or masquerading as product truth;
- repair against a stale product snapshot followed by export of a different snapshot.

Every accepted product mutation creates a new lineage state. Proof artifacts reference that exact state by immutable lineage identity/hash.

## 5. IR layers

Canonical IR layers are defined in `IR_TYPE_SYSTEM_V1.md`. The high-level flow is:

```text
ObservationEvidenceIR E
        ↓ deterministic geometric canonicalization
RiggingSurfaceIR S
        ↓ learned proposal
SkeletonProposalIR G*
        ↓ deterministic qualification / solver-backed normalization
QualifiedSkeletonIR G
        ↓ learned proposal
SkinProposalIR W*
        ↓ deterministic qualification / solver-backed normalization
QualifiedSkinIR W
        ↓ canonical assembly
CanonicalPuppetGraph Y
        ↓ derived, non-owning evidence
ProofFrame / FailureSignature / Attribution / RepairDirective
        ↓ accepted bounded mutation only
new CanonicalPuppetGraph Y'
```

The IR hierarchy deliberately separates **proposal**, **qualified state**, **canonical product state**, and **derived proof state**.

## 6. SurfaceBuilder boundary

SurfaceBuilder remains deterministic-first under the current evidence contract.

Candidate inputs:

- `P`: common/object-frame position evidence;
- `N`: orientation/normal evidence where qualified;
- view/pixel provenance;
- support/visibility evidence;
- cross-view persistence/cycle evidence;
- geometric uncertainty/risk;
- alpha/silhouette and image-grid continuity from the observation.

Candidate outputs in `RiggingSurfaceIR S`:

- oriented surfels/samples;
- canonical/common coordinates;
- local neighborhood graph;
- sheets/components;
- boundaries/seams/discontinuities;
- provenance/support;
- validity/uncertainty flags;
- optional deterministic view-local triangulations.

A learned topology head is **not canonical**. If deterministic topology is downstream-inferior, the next admissible intervention is local relational evidence plus constrained deterministic topology authority, not an opaque model-owned global topology.

## 7. Geppetto boundary

Geppetto owns learned rig proposal generation:

```text
RiggingSurfaceIR S → SkeletonProposalIR G*
```

`G*` may contain joint/control positions, proposed parents/hierarchy, confidence and surface support evidence. It does **not** own final IDs, DAG validity or product admission.

Skeleton qualification owns, at minimum:

- one admitted root where required;
- parent index validity;
- acyclicity;
- duplicate/helper cleanup policy;
- bounded structural completion where explicitly allowed;
- geometric support sanity;
- canonical ID assignment;
- fail-closed rejection when a valid editable graph cannot be justified.

The physical Geppetto architecture is still evidence-controlled; RigAnything-inspired autoregressive generation remains a strong candidate, not yet a promoted implementation authority.

## 8. Arachne boundary

Arachne owns learned skin proposal generation:

```text
RiggingSurfaceIR S + QualifiedSkeletonIR G → SkinProposalIR W*
```

`W*` may be dense, sparse or tokenized internally, but its external contract must permit deterministic qualification and editability.

Skin qualification owns, at minimum:

- non-negativity/bounds where required;
- row/simplex normalization;
- influence-count/sparsity policy;
- component/sheet leakage control;
- unsupported influence rejection;
- deterministic projection/repair where mathematically justified;
- residual reporting rather than silent correction.

SkinTokens-inspired generation remains a candidate implementation, not current architecture authority.

## 9. Compiler Core

Compiler Core is the only owner of canonical product state after learned proposals. It owns:

- canonical IDs and namespace;
- graph normalization and DAG invariants;
- admitted S/G/W bindings;
- deterministic rig/mesh/weight/deformation transforms;
- canonical product assembly;
- mutation/version lineage;
- qualification records;
- proof state binding;
- repair application;
- final serialization/export;
- fail-closed behavior.

The current product contract depicts Compiler after Arachne. This V2 skeleton makes typed qualification boundaries explicit without yet claiming that every qualification gate must execute as a continuously cross-cutting runtime control plane. Promotion of a fully cross-cutting Compiler execution plane remains evidence-controlled.

## 10. Solver-backed transforms

Solvers are not generic utilities. Each solver is a typed transformation or validator attached to explicit IR and invariants. The canonical matrix is in `SOLVER_AUTHORITY_MATRIX_V1.md`.

Recovered historical families that must be preserved/reconciled before rewriting include:

- constrained triangulation / CDT and cotangent geometry;
- rig graph/assembly optimization;
- BBW, active-set QP and KKT-family weight solvers;
- ARAP / posed ARAP / corrective deformation;
- XPBD, SDF and contact solvers;
- motion/deformation probe execution;
- failure attribution and bounded repair.

Historical numerical authority is not selected by timestamp. Exact source-file restoration remains pending forensic diff/dependency closure.

## 11. Proof / attribution / repair loop

```text
CanonicalPuppetGraph Y
        ↓
ResolvedMotionProbePlan
        ↓
MotionMeasurementReport
        ↓
MotionProofReport
     ┌───────┴────────┐
   PASS              FAIL
    │                  ↓
    │        MotionFailureSignatureSet
    │                  ↓
    │          OwnerAttributionReport
    │                  ↓
    │             RepairDirective
    │                  ↓
    │        bounded solver/proposal repair
    │                  ↓
    │       new CanonicalPuppetGraph Y'
    │                  ↓
    └──────────── re-probe ────────────┘
```

Proof and repair artifacts are **derived/non-owning**. They may reference canonical product nodes but cannot become a parallel product graph.

## 12. Runtime boundary

Historical runtime baseline to preserve:

```text
CanonicalPuppetGraph
        ↓
engine-neutral product serialization
        ↓
.rss / .realsas package
        + .rsr runtime payload
        ↓
C++17 runtime SDK
        ↓
stable C ABI + C++ wrapper
        ↓
Unity / Unreal / Godot / custom / WASM adapters
```

Unity is a host adapter, not canonical product ownership.

## 13. What is explicitly not restored

- old image-decomposition front-brain logic now owned by IRIS/SurfaceBuilder;
- authored-owner or teacher-exact mechanical ontology as a shipping perception requirement;
- source-rig exactness as the product objective;
- parallel product/proof truth;
- Unity-specific canonical ownership;
- god objects mixing product state, diagnostics, proof frames and historical runbooks;
- newer research solvers promoted merely because they are newer.

## 14. Evidence-controlled open boundaries

The following are **not frozen by this document**:

1. final IRIS head list beyond controlled substrate evidence;
2. whether SurfaceBuilder needs learned local relation evidence;
3. exact Geppetto neural architecture;
4. exact Arachne neural architecture;
5. whether intermediate Compiler qualification becomes a fully cross-cutting execution plane;
6. exact solver implementation selected from late-May vs Aug v0.5 historical lineages;
7. final runtime format evolution beyond preservation of the historical baseline.

## 15. Restoration sequence

```text
A. preserve this typed architecture skeleton
B. produce exact file-level historical restoration manifest
C. close dependency graph and byte/hash registry
D. freeze clean restoration source branch
E. source-diff late-May numerical kernels vs v0.5 ports
F. restore proof/repair/runtime core without old front-end ontology
G. write narrow adapters around frozen S/G/W types
H. run solver residual/invariant regression fixtures
I. run package → runtime ABI/end-to-end fixture
J. only then promote replacement implementations
```

No current IRIS training result is changed by this architecture document.
