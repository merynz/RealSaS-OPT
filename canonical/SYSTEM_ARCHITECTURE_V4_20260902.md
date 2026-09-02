# RealSaS — Canonical End-to-End System Architecture V4 — 2026-09-02

**Status:** `CANONICAL_V4_ARCHITECTURE_FROZEN__PRODUCT_V3_TYPED__SINGLE_FAMILY_E2E_BEFORE_GENERALIZATION`

**Supersedes for current composition:** `SYSTEM_ARCHITECTURE_V3_20260901.md`.

V3 remains historical authority for the audit state that led to this amendment. V4 closes the directional-renderable, appearance/completion, forest-root, capability and typed-proof gaps found by the final Drive/Library/GitHub adversarial audit.

## 1. Program order

Immediate objective:

> Prove one real clean 1024 eight-view family can travel end-to-end from shipping observations to a genuinely usable animated puppet: coherent skeleton, qualified skinning, deformable editable mesh, eight directional render states, source-attached appearance, at least one effective preset animation, exact proof and runtime projection.

**No generalization claim, held-out program or scale-up is authorized before this one-family E2E closure passes.**

## 2. Authority topology

```text
8 ordered RGBA observations + exact cameras
        |
        v
IRIS observation evidence
        |
        v
ObservationEvidenceIR
        |
        v
GeometricSubstrateAssembler
  P = O + dF
  support=True-only fusion
  qualified/hash-bound local geometry operators
        |
        v
RiggingSurfaceIR S
        |
        +--------------------+
        |                    |
        v                    v
Geppetto B_G             mesh/discretization
        |                    |
        v                    v
SkeletonProposalIR G*    M* per direction/component
        |                    |
        v                    v
Compiler graph           geometry qualification
        |                    |
        v                    v
QualifiedSkeletonIR.v2 G  QualifiedEditableMeshIR M
        |
        v
Arachne B_A
        |
        v
SkinProposalIR W*
        |
        v
Compiler skin qualification
        |
        v
QualifiedSkinIR W
        |
        +--------> mesh-weight binding -> QualifiedMeshSkinIR B

Shared mechanical state:
    S + G + W

Directional visual state:
    exact 8 x DirectionalRenderableIR
        each -> N RenderableComponentIR
             -> M + B
             -> per-face-corner AppearanceBindingIR authority
             -> zero or more QualifiedVisualCompletionIR

CapabilityContractIR + MotionStateIR
        |
        v
CanonicalPuppetGraph.v3
        |
        +------------------------------+
        |                              |
        v                              v
domain proof execution             bounded repair
        |                              |
        v                              |
ProductProofBundleIR <---------------+
        |
        | PASS exact same product hash
        v
RuntimePackageIR
```

Models produce evidence/proposals. Compiler owns canonical IDs, admissibility, product state and exact proof/export boundaries. Runtime is a projection only.

## 3. Mechanical state

`MechanicalStateIR` contains exactly S, `QualifiedSkeletonIR.v2` G, W and a mechanical-state hash.

### Surface support firewall

A persistence-group member with `support=False` may exist diagnostically but cannot contribute to fused `P`, raster bindings, support views, provenance or `SurfaceNode.source_observation_ids`. A group with zero supported members fails closed.

### Deterministic local geometry

IRIS learned geometric authority ends at forward depth/support/uncertainty. World `P` is analytic. Required normals/differentials are deterministic, qualified and operator-hash-bound derivatives of admitted S. Exact robust DTB-ND1 operator bytes remain a separate implementation/promotion item and may not be replaced by a weaker convenience estimator without parity evidence.

### Forest-safe skeleton

`QualifiedSkeletonIR.v2` stores `deform_root_ids[]` plus an optional separate `assembly_root_binding`. An assembly root, if present, must be explicitly non-deforming. The current restored optimizer may still emit one arborescence; that is an implementation state, not a product-schema restriction.

## 4. Directional visual state

`DirectionalRenderableSetIR` has **exact cardinality 8** and ordered views `0..7`.

Each `DirectionalRenderableIR` binds exact view/camera identity and at least one `RenderableComponentIR`. Component IDs and setup orders are unique inside a direction. Each component owns a direction-local qualified M and B, complete appearance coverage for every face corner, coverage class and zero or more qualified completion bindings. Shared S/G/W is never duplicated as eight mechanical truths.

## 5. Appearance authority — per face corner

Appearance authority is not component-global because a single renderable mesh may contain regions with different provenance.

Every mesh face corner must have exactly one `AppearanceCornerBinding` carrying:

- material UV used by runtime;
- donor view index;
- donor raster coordinate;
- source observation hash;
- authority class;
- optional completion ID;
- confidence.

Allowed authority classes:

- `OBSERVED_LOCAL` — donor is the target view;
- `OBSERVED_CROSS_VIEW` — donor is another truly observed view;
- `QUALIFIED_COMPLETION` — corner references an explicitly qualified completion ID.

Raster/provenance is authority; mesh-attached material coordinates are the deformation-stable compiled representation. Missing face-corner appearance is a product blocker.

## 6. Completion firewall

Unobserved completion cannot mutate or feed back into S/G/W.

```text
UNOBSERVED
 -> VisualCompletionProposalIR
 -> explicit qualification against observed S support
 -> QualifiedVisualCompletionIR
 -> referenced only by renderable appearance corners
```

Every completion proposal must be anchored to admitted observed surface IDs. A required completion that is not actually referenced by the renderable fails closed. A corner may not claim `QUALIFIED_COMPLETION` without referencing a qualified completion object.

## 7. Capability contract

Product identity contains capability policy, not proof results.

`CapabilityContractIR` records each capability as `REQUIRED`, `OPTIONAL` or `DISABLED`, plus implementation binding, policy hash and required proof domains.

The one-family E2E profile requires:

- `BASE_LBS`;
- `EDITABLE_MESH`;
- `QUALIFIED_SKINNING`;
- `VISUAL_8_DIRECTION`;
- `PRESET_MOTION`;
- `RUNTIME_BACKEND`.

Heavy ARAP, XPBD, contact and visual completion remain optional unless a product gate explicitly requires them. Historical `solver_registry.py` remains fail-closed provenance authority; a historical solver name is not executable capability proof.

## 8. Motion state — motion must be real

Static component order is setup state only.

`MotionStateIR` contains:

- `MotionClipIR[]`;
- typed `JointTransformTrackIR[]` over canonical joint IDs;
- optional `ComponentOrderTrackIR[]`;
- optional `ComponentVisibilityTrackIR[]`;
- exact motion-state hash.

`JointTransformTrackIR` contains time-ordered local translation / rotation / scale keys. If `PRESET_MOTION` is REQUIRED, the product must contain at least one PRESET clip with at least one canonical-joint track that changes transform state over time. An opaque clip hash or static track cannot satisfy the capability.

## 9. Canonical product identity

`CanonicalPuppetGraph.v3` binds:

- `mechanical_state_hash`;
- `directional_visual_state_hash`;
- `capability_contract_hash`;
- `motion_state_hash`;
- qualification ledger;
- editable metadata;
- runtime policy;
- parent state hash.

Any accepted mutation to mechanical truth, directional mesh/weights/appearance/completion, capability policy or motion state creates a different `product_state_hash`.

## 10. Proof is sibling-derived state

Proof results are deliberately **not** hashed into the product they prove.

V4 decomposes proof into `ProofPlanIR`, `MeasurementReportIR`, `DomainProofReportIR`, `ProductProofBundleIR`, plus optional `CapabilityQualificationIR`.

Required one-family E2E domains are derived from capability policy and include `MECHANICAL_STRUCTURE`, `MESH_QUALITY`, `DEFORMATION`, `DIRECTIONAL_VISUAL`, `MOTION`, `RUNTIME_CONSUMPTION`.

Every report is bound to the exact product hash. Product mutation invalidates prior proof. Missing required domains produce `ABSTAIN`; a required-domain failure produces `FAIL`; all required domains PASS produces product-proof PASS.

## 11. Repair and runtime

Historical v0.5 proof/failure/owner-attribution/repair semantics remain restoration authority:

```text
Y -> proof -> failure signatures -> owner attribution
  -> bounded RepairDirective -> Y' or ABSTAIN -> mandatory re-proof
```

Repair may not silently mutate a product while preserving its old hash.

`RuntimePackageIR` may only be projected from exact `CanonicalPuppetGraph.v3` plus a matching `ProductProofBundleIR.overall_status == PASS`. Runtime records source product and proof-bundle hashes and never becomes product truth.

## 12. Implementation classes

### Canonical executable / typed now

- support=True-only surface fusion;
- existing S/W/M/B qualification primitives;
- `QualifiedSkeletonIR.v2` product-facing type;
- `MechanicalStateIR`;
- per-corner appearance authority and completion firewall;
- exact-eight directional renderables;
- capability contract;
- typed canonical-joint preset motion tracks;
- `CanonicalPuppetGraph.v3`;
- typed proof bundle and stale-proof rejection;
- direction/component/track-safe bundle routing.

### Typed but implementation-partial

- true multi-root graph production;
- robust DTB-ND1 operator byte promotion;
- production MWB-2 topology/interpolation;
- Geppetto candidate model;
- SkinFieldCodec;
- Arachne predictor;
- actual appearance compiler from eight source rasters;
- heavy historical proof/repair execution;
- actual LBS/deformation execution over V3 renderables;
- native runtime materialization.

### Rejected from core

- old FrontBrain/semantic-slot ownership;
- teacher-exact product identity;
- one final M/B for all eight directions;
- hidden completion as mechanical truth;
- hidden BBW semantic skin generation;
- proof results inside product identity;
- opaque/static clip metadata counting as animation capability.

## 13. Next gate — SINGLE-FAMILY E2E FIT

V4 is considered product-valid only when one clean real family proves:

```text
8x1024 observations
 -> IRIS -> S
 -> Geppetto -> G
 -> Arachne/codec -> W
 -> directional M/B/appearance
 -> effective preset joint animation
 -> exact deformation/render proof
 -> PASS RuntimePackageIR
```

Acceptance means a visibly coherent character with a sensible skeleton, usable skinning, stable deformation and a preset animation that genuinely moves the compiled puppet. Teacher truth may be used for training/evaluation only and may not cross the shipping inference boundary.

Only after this PASS does the program open heterogeneous FIT8/generalization work.
