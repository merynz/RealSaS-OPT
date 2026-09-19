# RealSaS — Qualified Mesh + Presentation Authority Contract V1 — 2026-09-18

**Status:** ARCHITECTURE_FROZEN__MAINLINE_PLAN_REBASED__IMPLEMENTATION_STARTED  
**Scope:** subject-agnostic product architecture; Knight is only the next witness  
**Claim boundary:** this freezes authority and gate structure. It is not a Knight PASS and does not widen historical Mage evidence.

## Frozen authority decisions

1. **QualifiedMeshIR is the single canonical product geometry authority.** `RiggingSurfaceIR S` remains the admitted mechanical/evidence substrate. Mesh producers (CDT today, another backend later) produce candidates only. Runtime, appearance and motion may consume a QualifiedMesh; they may not create a second topology truth.
2. **MechanicalPartitionIR does not mutate S.** It assigns admitted S nodes to structural components and emits boundary constraints: `SEPARATE`, `PRESERVE_CONTINUITY`, or `UNKNOWN`. The mesh producer performs topology operations; qualification proves the constraints were respected.
3. **Mesh production is view-independent.** The producer interface is conceptually `(S, MechanicalPartitionIR, MeshPolicy) -> CanonicalMeshCandidateIR`. View/camera identity does not belong to canonical product mesh authority. Historical view-local CDT remains numerical evidence/donor machinery, not the new authority contract.
4. **QualifiedPresentationGraphIR owns automatic Spine-class presentation semantics.** It binds independently keyable slots, attachments, mechanical behavior, carrier class, eight view overlays, appearance/composition bindings and evidence provenance. It is subordinate to one exact `CanonicalPuppetStateIR`: the graph binds that product-state hash, its slot bones must resolve to the exact qualified skeleton, its non-CLIP attachment component references must cover the exact mechanical partition, MESH carriers bind the exact QualifiedMesh lineage, and its eight overlay camera hashes must equal the frozen deformation-envelope camera set. It does not require categorical object identity such as “sword”.

## Boundary semantics

`SEPARATE` requires a real product-mesh separation.  
`PRESERVE_CONTINUITY` forbids a semantic cut.  
`UNKNOWN` may be provisionally preserved by a producer, but qualification may not silently promote that choice to truth. If the boundary is consequential within the declared deformation capability envelope, QualifiedMesh must FAIL. A changed partition is a new lineage and requires a new candidate.

Numerical chart seams never gain semantic boundary authority. A chart seam inside a `PRESERVE_CONTINUITY` region requires exact stitch/equivalence proof under G2.

## G1–G5 QualifiedMesh contract

The five gates are independent. No aggregate score can compensate for a failed gate.

- **G1 — support and lineage:** every product vertex has admitted S support; unsupported/cross-component support is forbidden. Optional dense-surface refinement is a bounded correction from the S-derived base point, with exact dense lineage. Projection bounds are local-scale-relative; tangential drift is separately bounded and component membership may not change.
- **G2 — declared topology integrity:** zero degenerate/duplicate faces, no undeclared cracks/T-junctions/zero-area connections, and chart seams satisfy declared continuity. Global 2-manifoldness is not itself a product requirement; declared topology consistency is.
- **G3 — deformation conditioning:** rest quality plus stress probes under the exact `DeformationCapabilityEnvelopeIR`; no inversion/foldover and bounded area, edge and aspect degradation. Historical FIT2 `0.25° / 250` remains a scientific-surface compatibility floor, not animation-grade admission. Current G3 additionally requires the frozen envelope stress probe to keep local triangle area ratio within `[0.05,20]` and local deformation condition number `<=16`; these are numerical conditioning bounds, not a professional-motion aesthetic score.
- **G4 — component-boundary compliance:** `SEPARATE` and `PRESERVE_CONTINUITY` constraints are obeyed. Consequential UNKNOWN boundaries are forbidden at PASS.
- **G5 — multiview visible coverage:** exact canonical M is projected into all eight authoritative cameras and rasterized once per view with the Runtime-v3/v4 half-integer TOP_LEFT rule plus camera-forward z-buffer visibility. Source component masks must be a disjoint, complete partition of the exact stage-07 qualified source foreground; their source-observation and camera hashes must match `QualifiedObservationSetIR`. Coverage is evaluated on z-visible component-owner masks as a complete `(view, admitted component, carrier class)` matrix, with recall/precision and largest coherent hole as first-class measurements. A global score may not hide a missing component or an occlusion-owner swap.

Carrier-class policy must be frozen before candidate qualification. `ComponentCarrierPolicyIR` binds exactly one product-geometry carrier class (`MESH` or `PLANAR`) to every admitted component and is hash-bound by the candidate, QualifiedMesh and QualifiedPresentationGraph. `CLIP` is presentation-mask semantics, not a mechanical component carrier, and therefore cannot be used to escape mesh coverage qualification. A failed MESH candidate may report `POSSIBLE_CARRIER_MISCLASSIFICATION` only as a diagnostic; reclassification to PLANAR/CLIP creates a new carrier-policy lineage, candidate and qualification lineage.

`MeshQualificationPolicyIR` is also first-class rather than an opaque hash string. It binds G1 refinement limits, a G3 rest-conditioning policy that may be stricter but never weaker than the subject-free calibrated 7.5° / aspect-16 numerical floor, and per-carrier G5 recall/precision/coherent-hole/interior-uncovered thresholds. `interior_uncovered_fraction` is a hard anti-peppering metric and cannot be compensated by aggregate recall. QualifiedMesh validation consumes the exact policy object and independently applies those thresholds; a report-level PASS flag cannot weaken them.

## DeformationCapabilityEnvelopeIR

The envelope is a numerical conditioning-policy artifact used by:
- G3 mesh stress qualification,
- consequential-UNKNOWN analysis,
- UNSEEN exposure qualification,
- Stage35 diagnostics/proof lineage.

It binds exact skeleton identity, a frozen 3D joint-axis contract, bounded micro-stress transforms, exact eight-camera set, allowed attachment state space and an exact probe-plan hash. It is not a third product geometry authority and it is **not** the kinematic range or admissibility authority for professional artist clips. Stage25/27 prove bounded numerical mesh/skin conditioning only. Professional motion capability is minted only when Stage35 executes the exact Stage34 full-3D local-quaternion clip on the exact QualifiedMeshIR + QualifiedMeshSkinIR and passes the dynamic/contact/visibility gates.

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

The metric set and authority structure above were frozen before Knight product results. Animation-grade numerical thresholds are intentionally **not copied from the pathological historical Mage mesh distribution**. The subject-free calibration preregistration and result are sealed in `canonical/ANIMATION_GRADE_MESH_CONDITIONING_CALIBRATION_PREREG_20260918.md` and `canonical/ANIMATION_GRADE_MESH_CONDITIONING_CALIBRATION_RESULT_20260918.json`; the current admission values are frozen in `canonical/QUALIFIED_MESH_PRODUCT_POLICY_V1_20260918.json`. None of those artifacts claims Knight compliance.

## Mainline execution binding

Stages 24–30 are now executable typed adapters: `MECHANICAL_PARTITION_QUALIFIED -> DEFORMATION_CAPABILITY_ENVELOPE -> MESH_CANDIDATE_BUILD -> QUALIFIED_MESH_GATE -> QUALIFIED_MESH_SKIN_TRANSFER -> CANONICAL_PUPPET_STATE_SEALED -> QUALIFIED_PRESENTATION_GRAPH`. Stage 30 is an aggregate over three explicit subordinate authorities: role-free `QualifiedPresentationStructureIR`, canonical-mesh eight-view `QualifiedAppearanceSetIR`, and z-visible `QualifiedCompositionSetIR`. Exact source-raster byte identity is bound by `QualifiedObservationSetIR`; one donor view is required per face and cross-view color blending is forbidden. Automatic PLANAR proxy production and visual completion remain fail-closed rather than implicit. Dependency edges are exact consumed-authority edges and the orchestrator invalidates only the changed stage plus its transitive dependency subgraph. Executable adapter status is not a Knight PASS; the active run remains 0/40 until exact run artifacts pass these stages.


## Rest render authority

Stage 05 now seals the exact eight-camera parameter set as `QualifiedCameraSetIR`; envelope, G5 and rest rendering consume that same artifact. Stage 31 is a deterministic consumer, not a new product-authority owner: it projects the exact canonical `QualifiedMeshIR`, resolves physical visibility with the same z-buffer/tie semantics as G5, and transports exact source RGBA through the qualified one-donor-per-face appearance bindings. Lighting, material shading, relighting and completion are forbidden. `RestRenderSetIR` binds raw rendered RGBA hashes plus visible donor provenance counts; PNG files are evidence transport only. Stage 32 remains a separate source-preservation gate and may not select thresholds from Knight results.
