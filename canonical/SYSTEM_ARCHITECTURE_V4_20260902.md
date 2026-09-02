# RealSaS — Canonical End-to-End System Architecture V4 — 2026-09-02

**Status:** `CANONICAL_V4_ARCHITECTURE_FROZEN__3D_EQUIVALENT_MECHANICS__DIRECTIONAL_2D_2P5D_PRODUCT__SINGLE_FAMILY_E2E_BEFORE_GENERALIZATION`

**Supersedes for current composition:** `SYSTEM_ARCHITECTURE_V3_20260901.md`.

V3 remains historical authority for the audit state that led to this amendment. V4 closes the directional-renderable, appearance/completion, forest-root, capability and typed-proof gaps found by the final Drive/Library/GitHub adversarial audit.

## 1. Product ontology — binding

RealSaS is **not a full 3D reconstruction system**.

The architectural target is a **3D-equivalent mechanical representation** compiled into an editable eight-direction 2D/2.5D puppet.

The intended equivalence is downstream/mechanical, not geometric identity:

```text
f_rig(S_hat) ~= f_rig(S_3D)
```

where `S_hat` is the observation-grounded RealSaS mechanical substrate and `S_3D` is the kind of complete 3D surface a 3D rigging system could consume. RealSaS must preserve enough mechanics for skeleton, skinning and deformation decisions to be functionally equivalent for the product task. It is **not required** to recover a watertight body, exact hidden cavities, exact thickness, a canonical volumetric scene, or a true 3D runtime character.

Binding invariant:

`3D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`.

World/camera-space `P`, depth, normals/differentials and multi-view correspondence are evidence/conditioning carriers. They may be 3-component geometric quantities without becoming authoritative product-level 3D reconstruction.

Shipping product:

`8 directional 2D/2.5D renderables + shared mechanical control identity + qualified skin/deformation semantics`.

## 2. Program order

Immediate objective:

> Prove one real clean 1024 eight-view family can travel end-to-end from shipping observations to a genuinely usable animated puppet: coherent skeleton, qualified skinning, deformable editable directional meshes, source-attached appearance, at least one effective preset animation, exact proof and runtime projection.

**No generalization claim, held-out program or scale-up is authorized before this one-family E2E closure passes.**

## 3. Authority topology

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
RiggingSurfaceIR S_hat
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
    S_hat + G + W

Directional product state:
    exact 8 x DirectionalRenderableIR
        each -> N RenderableComponentIR
             -> direction-local M + B
             -> per-face-corner AppearanceBindingIR
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

## 4. Mechanical state

`MechanicalStateIR` contains exactly `S_hat`, `QualifiedSkeletonIR.v2` G, W and a mechanical-state hash.

`S_hat` is a mechanical substrate, not a reconstructed closed character mesh.

### Surface support firewall

A persistence-group member with `support=False` may exist diagnostically but cannot contribute to fused `P`, raster bindings, support views, provenance or `SurfaceNode.source_observation_ids`. A group with zero supported members fails closed.

### Deterministic local geometry

IRIS learned geometric authority ends at forward depth/support/uncertainty. World `P` is analytic. Required normals/differentials are deterministic, qualified and operator-hash-bound derivatives of admitted S. Exact robust DTB-ND1 operator bytes remain a separate implementation/promotion item and may not be replaced by a weaker convenience estimator without parity evidence.

These operators support mechanical inference; they do not promote S into full-3D reconstruction truth.

### Forest-safe skeleton

`QualifiedSkeletonIR.v2` stores `deform_root_ids[]` plus an optional separate `assembly_root_binding`. An assembly root, if present, must be explicitly non-deforming. The current restored optimizer may still emit one arborescence; that is an implementation state, not a product-schema restriction.

Skeleton positions may use the shared mechanical carrier internally, but runtime animation semantics remain directional 2D/2.5D puppet deformation rather than a claim of a fully reconstructed 3D rig.

## 5. Directional visual state

`DirectionalRenderableSetIR` has **exact cardinality 8** and ordered views `0..7`.

Each `DirectionalRenderableIR` binds exact view/camera identity and at least one `RenderableComponentIR`. Component IDs and setup orders are unique inside a direction. Each component owns a direction-local qualified M and B, complete appearance coverage for every face corner, coverage class and zero or more qualified completion bindings.

Shared mechanical identity does not mean a single reconstructed 3D render mesh. The product is explicitly direction-local in render geometry/appearance.

## 6. Appearance authority — per face corner

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

## 7. Completion firewall

Unobserved completion cannot mutate or feed back into `S_hat/G/W`.

```text
UNOBSERVED
 -> VisualCompletionProposalIR
 -> explicit qualification against observed S support
 -> QualifiedVisualCompletionIR
 -> referenced only by renderable appearance corners
```

Completion is visual/product completion only; it is not hidden recovered 3D geometry.

## 8. Capability contract

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

## 9. Motion state — directional 2D/2.5D deformation

`MotionStateIR` contains:

- `MotionClipIR[]`;
- typed `JointTransformTrackIR[]` over canonical joint IDs;
- optional `ComponentOrderTrackIR[]`;
- optional `ComponentVisibilityTrackIR[]`;
- exact motion-state hash.

A `JointTransformKeyIR` is a **puppet-local 2D/2.5D deformation key**:

- `translation_xy`;
- in-plane `rotation_deg`;
- `scale_xy`;
- optional `depth_offset` used only for ordering/deformer context.

It is intentionally not a 3D rigid transform or quaternion pose.

If `PRESET_MOTION` is REQUIRED, at least one PRESET clip must contain a canonical-joint track that changes puppet-local state over time. Opaque clip metadata or a static track cannot satisfy the capability.

## 10. Canonical product identity

`CanonicalPuppetGraph.v3` binds:

- `mechanical_state_hash`;
- `directional_visual_state_hash`;
- `capability_contract_hash`;
- `motion_state_hash`;
- qualification ledger;
- editable metadata;
- runtime policy;
- parent state hash.

The product declares representation `DIRECTIONAL_2D_2P5D_PUPPET` and `full_3d_reconstruction_authority=false`.

Any accepted mutation to mechanical truth, directional mesh/weights/appearance/completion, capability policy or motion state creates a different `product_state_hash`.

## 11. Proof is sibling-derived state

Proof results are deliberately **not** hashed into the product they prove.

V4 decomposes proof into `ProofPlanIR`, `MeasurementReportIR`, `DomainProofReportIR`, `ProductProofBundleIR`, plus optional `CapabilityQualificationIR`.

Required one-family E2E domains are derived from capability policy and include `MECHANICAL_STRUCTURE`, `MESH_QUALITY`, `DEFORMATION`, `DIRECTIONAL_VISUAL`, `MOTION`, `RUNTIME_CONSUMPTION`.

Every report is bound to the exact product hash. Product mutation invalidates prior proof. Missing required domains produce `ABSTAIN`; a required-domain failure produces `FAIL`; all required domains PASS produces product-proof PASS.

Proof asks whether the 3D-equivalent mechanical evidence compiles into correct product behavior. It does not ask whether the system recovered the unique true 3D character.

## 12. Repair and runtime

Historical proof/failure/owner-attribution/repair semantics remain restoration authority:

```text
Y -> proof -> failure signatures -> owner attribution
  -> bounded RepairDirective -> Y' or ABSTAIN -> mandatory re-proof
```

Repair may not silently mutate a product while preserving its old hash.

`RuntimePackageIR` may only be projected from exact `CanonicalPuppetGraph.v3` plus a matching `ProductProofBundleIR.overall_status == PASS`. Runtime records source product and proof-bundle hashes and declares the directional 2D/2.5D representation. Runtime never becomes a second product truth or a hidden 3D reconstruction.

## 13. Implementation classes

### Canonical executable / typed now

- support=True-only surface fusion;
- existing S/W/M/B qualification primitives;
- `QualifiedSkeletonIR.v2` product-facing type;
- `MechanicalStateIR`;
- per-corner appearance authority and completion firewall;
- exact-eight directional renderables;
- capability contract;
- typed 2D/2.5D canonical-joint preset motion tracks;
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
- actual directional LBS/deformation execution over V3 renderables;
- native runtime materialization.

### Rejected from core

- full 3D reconstruction as product authority;
- watertight/full hidden geometry requirement;
- 3D rigid-body/quaternion motion as canonical puppet requirement;
- old FrontBrain/semantic-slot ownership;
- teacher-exact product identity;
- one final M/B for all eight directions;
- hidden completion as mechanical truth;
- hidden BBW semantic skin generation;
- proof results inside product identity;
- opaque/static clip metadata counting as animation capability.

## 14. Next gate — SINGLE-FAMILY E2E FIT

V4 is considered product-valid only when one clean real family proves:

```text
8x1024 observations
 -> IRIS -> S_hat (3D-equivalent mechanical substrate)
 -> Geppetto -> G
 -> Arachne/codec -> W
 -> 8 directional M/B/appearance
 -> effective 2D/2.5D preset joint animation
 -> exact deformation/render proof
 -> PASS RuntimePackageIR
```

Acceptance means a visibly coherent character with a sensible skeleton, usable skinning, stable directional deformation and a preset animation that genuinely moves the compiled puppet. Teacher truth may be used for training/evaluation only and may not cross the shipping inference boundary.

Only after this PASS does the program open heterogeneous FIT8/generalization work.
