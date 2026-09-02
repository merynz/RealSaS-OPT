# RealSaS — Canonical End-to-End System Architecture V4 — 2026-09-02

**Status:** `CANONICAL_V4_ARCHITECTURE_FROZEN__PRODUCT_V3_TYPED__SINGLE_FAMILY_E2E_BEFORE_GENERALIZATION`

**Supersedes for current composition:** `SYSTEM_ARCHITECTURE_V3_20260901.md`.

V3 remains historical authority for the audit state that led to this amendment. V4 closes the directional-renderable, appearance/completion, forest-root, capability and typed-proof gaps found by the final Drive/Library/GitHub adversarial audit.

## 1. Program order

The immediate scientific/product objective is deliberately narrow:

> First prove one real 1024 eight-view family can travel end-to-end from shipping observations to a genuinely usable animated puppet: coherent skeleton, qualified skinning, deformable editable mesh, eight directional render states, one preset animation, exact proof and runtime projection.

**No generalization claim, held-out claim or large-scale training program is authorized before this one-family E2E closure passes.**

After one-family E2E PASS, the next program becomes heterogeneous FIT8/generalization.

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
             -> AppearanceBindingIR
             -> optional QualifiedVisualCompletionIR

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

IRIS learned geometric authority ends at forward depth/support/uncertainty. World `P` is analytic. Required normals/differentials are deterministic, qualified and operator-hash-bound derivatives of admitted S. V4 types the boundary; exact robust DTB-ND1 operator bytes remain a separate implementation/promotion item and may not be replaced by a weaker convenience estimator without parity evidence.

### Forest-safe skeleton

`QualifiedSkeletonIR.v2` stores `deform_root_ids[]` plus an optional separate `assembly_root_binding`. An assembly root, if present, must be explicitly non-deforming. The current restored optimizer may still emit one arborescence; that is an implementation state, not a product-schema restriction.

## 4. Directional visual state

`DirectionalRenderableSetIR` has **exact cardinality 8** and ordered views `0..7`.

Each `DirectionalRenderableIR` binds exact view/camera identity and N `RenderableComponentIR` values. Each component owns a direction-local qualified M and B, an `AppearanceBindingIR`, setup order, coverage class and optional qualified completion. Shared S/G/W is never duplicated as eight mechanical truths.

## 5. Appearance authority

Shipping appearance is mesh-attached but observation-derived.

`AppearanceBindingIR.authority_class` is one of `OBSERVED_LOCAL`, `OBSERVED_CROSS_VIEW`, `QUALIFIED_COMPLETION`.

Per-corner bindings carry mesh-attached material UV, donor view, donor raster coordinate, source observation hash and confidence. Raster/provenance is authority; runtime material coordinates are the deformation-stable compiled representation. `UNKNOWN` is not admitted as product appearance truth.

## 6. Completion firewall

Unobserved completion cannot mutate or feed back into S/G/W.

```text
UNOBSERVED
 -> VisualCompletionProposalIR
 -> explicit qualification
 -> QualifiedVisualCompletionIR
 -> renderable component only
```

A completion proposal must be anchored to admitted observed surface support. If a required completion cannot be qualified, the required product capability fails or abstains. Completion is never a hidden second geometry, skeleton or skin authority.

## 7. Capability contract

Product identity contains capability policy, not proof results.

`CapabilityContractIR` records each capability as `REQUIRED`, `OPTIONAL` or `DISABLED`, plus implementation binding, policy hash and required proof domains.

The one-family E2E profile requires `BASE_LBS`, `EDITABLE_MESH`, `QUALIFIED_SKINNING`, `VISUAL_8_DIRECTION`, `PRESET_MOTION`, `RUNTIME_BACKEND`.

Heavy ARAP, XPBD, contact and visual completion remain optional unless a product gate explicitly requires them. Historical `solver_registry.py` remains fail-closed provenance authority; a historical solver name is not executable capability proof.

## 8. Motion state

Static component order is setup state only. `MotionStateIR` contains clips plus optional component order/visibility tracks and a motion-state hash. The first E2E closure requires at least one `PRESET` clip. Learned animation generation is not required.

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

Historical v0.5 proof/failure/owner-attribution/repair semantics remain the restoration authority:

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
- appearance binding and completion firewall types;
- exact-eight directional renderables;
- capability contract and motion state;
- `CanonicalPuppetGraph.v3`;
- typed proof bundle and stale-proof rejection;
- direction/component-safe bundle routing.

### Typed but implementation-partial

- true multi-root graph production;
- robust DTB-ND1 operator byte promotion;
- production MWB-2 topology/interpolation;
- Geppetto candidate model;
- SkinFieldCodec;
- Arachne predictor;
- heavy historical proof/repair execution;
- native runtime materialization.

### Rejected from core

- old FrontBrain/semantic-slot ownership;
- teacher-exact product identity;
- one final M/B for all eight directions;
- hidden completion as mechanical truth;
- hidden BBW semantic skin generation;
- proof results inside product identity.

## 13. Next gate — SINGLE-FAMILY E2E FIT

V4 is considered useful only when one clean real family proves:

```text
8x1024 observations
 -> IRIS -> S
 -> Geppetto -> G
 -> Arachne/codec -> W
 -> directional M/B/appearance
 -> preset animation
 -> exact deformation/render proof
 -> PASS RuntimePackageIR
```

Acceptance means a visibly coherent character with a sensible skeleton, usable skinning, stable deformation and a preset animation that genuinely moves the compiled puppet. Teacher truth may be used for training/evaluation only and may not cross the shipping inference boundary.

Only after this PASS does the program open heterogeneous FIT8/generalization work.
