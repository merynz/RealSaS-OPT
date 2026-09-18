# RealSaS — Qualified Mesh + Presentation Authority Contract V1 — 2026-09-18

**Status:** ARCHITECTURE_FROZEN__MAINLINE_PLAN_REBASED__IMPLEMENTATION_STARTED  
**Scope:** subject-agnostic product architecture; Knight is only the next witness  
**Claim boundary:** this freezes authority and gate structure. It is not a Knight PASS and does not widen historical Mage evidence.

## Frozen authority decisions

1. **QualifiedMeshIR is the single canonical product geometry authority.** `RiggingSurfaceIR S` remains the admitted mechanical/evidence substrate. Mesh producers (CDT today, another backend later) produce candidates only. Runtime, appearance and motion may consume a QualifiedMesh; they may not create a second topology truth.
2. **MechanicalPartitionIR does not mutate S.** It assigns admitted S nodes to structural components and emits boundary constraints: `SEPARATE`, `PRESERVE_CONTINUITY`, or `UNKNOWN`. The mesh producer performs topology operations; qualification proves the constraints were respected.
3. **Mesh production is view-independent.** The producer interface is conceptually `(S, MechanicalPartitionIR, MeshPolicy) -> CanonicalMeshCandidateIR`. View/camera identity does not belong to canonical product mesh authority. Historical view-local CDT remains numerical evidence/donor machinery, not the new authority contract.
4. **QualifiedPresentationGraphIR owns automatic Spine-class presentation semantics.** It binds independently keyable slots, attachments, mechanical behavior, carrier class, eight view overlays, appearance/composition bindings and evidence provenance. It does not require categorical object identity such as “sword”.

## Boundary semantics

`SEPARATE` requires a real product-mesh separation.  
`PRESERVE_CONTINUITY` forbids a semantic cut.  
`UNKNOWN` may be provisionally preserved by a producer, but qualification may not silently promote that choice to truth. If the boundary is consequential within the declared deformation capability envelope, QualifiedMesh must FAIL. A changed partition is a new lineage and requires a new candidate.

Numerical chart seams never gain semantic boundary authority. A chart seam inside a `PRESERVE_CONTINUITY` region requires exact stitch/equivalence proof under G2.

## G1–G5 QualifiedMesh contract

The five gates are independent. No aggregate score can compensate for a failed gate.

- **G1 — support and lineage:** every product vertex has admitted S support; unsupported/cross-component support is forbidden. Optional dense-surface refinement is a bounded correction from the S-derived base point, with exact dense lineage. Projection bounds are local-scale-relative; tangential drift is separately bounded and component membership may not change.
- **G2 — declared topology integrity:** zero degenerate/duplicate faces, no undeclared cracks/T-junctions/zero-area connections, and chart seams satisfy declared continuity. Global 2-manifoldness is not itself a product requirement; declared topology consistency is.
- **G3 — deformation conditioning:** rest quality plus stress probes under the exact `DeformationCapabilityEnvelopeIR`; no inversion/foldover and bounded area, edge and aspect degradation. Historical FIT2 `0.25° / 250` remains a scientific-surface compatibility floor, not animation-grade admission.
- **G4 — component-boundary compliance:** `SEPARATE` and `PRESERVE_CONTINUITY` constraints are obeyed. Consequential UNKNOWN boundaries are forbidden at PASS.
- **G5 — multiview visible coverage:** exact canonical M is projected into all eight authoritative cameras. Coverage is evaluated as a complete `(view, admitted component, carrier class)` matrix, with recall/precision and largest coherent hole as first-class measurements. A global score may not hide a missing component.

Carrier-class policy must be frozen before candidate qualification. `ComponentCarrierPolicyIR` binds exactly one carrier class to every admitted component and is hash-bound by the candidate, QualifiedMesh and QualifiedPresentationGraph. A failed MESH candidate may report `POSSIBLE_CARRIER_MISCLASSIFICATION` only as a diagnostic; reclassification to PLANAR/CLIP creates a new carrier-policy lineage, candidate and qualification lineage.

`MeshQualificationPolicyIR` is also first-class rather than an opaque hash string. It binds G1 refinement limits, a G3 rest-conditioning policy that may be stricter but never weaker than the subject-free calibrated 7.5° / aspect-16 numerical floor, and per-carrier G5 recall/precision/coherent-hole thresholds. QualifiedMesh validation consumes the exact policy object and independently applies those thresholds; a report-level PASS flag cannot weaken them.

## DeformationCapabilityEnvelopeIR

The envelope is a qualification-policy artifact shared by:
- G3 mesh stress qualification,
- consequential-UNKNOWN analysis,
- UNSEEN exposure qualification,
- MotionProof clip admissibility.

It binds exact skeleton identity, bounded joint transform ranges, exact eight-camera set, allowed attachment state space and an exact probe-plan hash. It is not a third product geometry authority.

## Structural vs presentation segmentation

RealSaS does not require categorical object recognition for product correctness.

- structural/mechanical partition asks where the puppet must or may separate;
- presentation segmentation asks which regions need independent addressing for attachment selection, tint, clipping, order, visibility or editability;
- categorical recognition (“this is a sword”) is not product authority.

Generic appearance boundaries may resolve structural ambiguity but may not introduce categorical object identity.

## Presentation evidence classes

Presentation decisions retain their evidence class:
- mechanical structure → slot/attachment identity and mechanical binding;
- rest observations → setup composition / clipping evidence;
- posed geometry → physical depth/occlusion;
- motion override → intentional animated order/visibility changes;
- source appearance → UV/donor/provenance.

Depth and slot order must not independently solve the same physical occlusion. Volumetric/deformable geometry uses posed depth. PLANAR carriers use a qualified posed 3D proxy plane (not a single scalar depth); semantic order remains tie-break/clipping/intentional override authority.

## Numeric threshold status

The metric set and authority structure above are frozen before Knight product results. Animation-grade numerical thresholds are intentionally **not copied from the pathological historical Mage mesh distribution**. They must be sealed in a separate preregistration from controlled synthetic conditioning sweeps and solver behavior before any Knight QualifiedMesh result is inspected.

Until that preregistration exists, no new canonical QualifiedMesh product PASS may be claimed.

## Mainline execution binding

Stages 24–30 are now reserved as: `MECHANICAL_PARTITION_QUALIFIED -> DEFORMATION_CAPABILITY_ENVELOPE -> MESH_CANDIDATE_BUILD -> QUALIFIED_MESH_GATE -> QUALIFIED_MESH_SKIN_TRANSFER -> CANONICAL_PUPPET_STATE_SEALED -> QUALIFIED_PRESENTATION_GRAPH`. All remain fail-closed and `UNBOUND` until their adapters satisfy this contract; the plan rename itself is not implementation PASS.
