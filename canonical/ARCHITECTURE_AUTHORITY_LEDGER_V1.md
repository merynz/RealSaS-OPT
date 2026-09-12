# RealSaS — Architecture Authority Ledger V1

**Updated:** 2026-09-12  
**Repository-wide continuation authority:** `CURRENT_STATE.md` on `main`  
**Active executable refit branch:** `repair/mage-full-subject-reclosure-20260912`

This ledger distinguishes implementation, executed evidence, promotion, product authority and generalization.

## Current authority matrix

| ID | Mechanism / responsibility | Status | Current interpretation |
|---|---|---|---|
| `SYS_V4_2D_PUPPET_TARGET` | 8-direction editable 2D/2.5D puppet | **BINDING** | artist-visible directional deformation product, not full-3D reconstruction |
| `H1_OBSERVABLE_PRODUCT_SURFACE` | observation-conditioned geometry/support evidence | **CURRENT CORRECTED MAGE AUTHORITY / CLOSED** | exact eight-view product observation authority; hidden/global watertight teacher sign is diagnostic only |
| `GSA_RIGGING_SURFACE_ASSEMBLY` | admitted evidence -> `RiggingSurfaceIR` | **BINDING / CURRENT GSA8192 PASS** | deterministic provenance/geometry/raster-support authority; 8171 nodes / 23656 relations |
| `GEPPETTO_REFERENCE_STRENGTH_V1` | skeleton/control proposal | **HISTORICAL FIT1 FROZEN; FRESH FIT2 ACTIVE** | old checkpoint preserved but forbidden to load; corrected-lineage fresh refit is running |
| `COMPILER_EXACT_GRAPH_LEGALITY` | final legal root/parent/tree/canonical IDs | **BINDING** | no hidden semantic repair |
| `SKIN_FIELD_CODEC_A0` | continuous skin-field representation/oracle lineage | **RETAINED RESEARCH / REPRESENTATION PROOF** | historical K4 representability evidence; no V5 runtime dependency |
| `ARACHNE_A1_V4_BACKBONE` | rich qualified surface+skeleton -> K4-Z | **HISTORICAL FIT1 BACKBONE EVIDENCE** | old frozen-decoder route failed; direct readout proved representation sufficient on old witness |
| `ARACHNE_DIRECT_NXJ_SIMPLEX` | row-wise joint competition | **HISTORICAL EXECUTED CAUSAL PASS** | old-lineage H and Z direct heads passed; not corrected FIT2 authority |
| `ARACHNE_V5_MINIMAL_K4_DIRECT_SIMPLEX` | K4-Z -> direct row-simplex `SkinProposalIR` | **HISTORICAL FIT1 PROMOTED; FRESH FIT2 PENDING** | old-S V5 closure preserved; corrected-lineage Arachne blocked on Geppetto |
| `COMPILER_SKIN_QUALIFICATION` | skin reference/simplex legality | **BINDING** | Compiler owns qualified row legality, not missing skin semantics |
| `DIRECTIONAL_ALPHA_DOMAIN_MESH` | artist-visible view-local deformation/render topology | **FIRST-CLASS PRODUCT AUTHORITY** | exact qualified mesh is the physical domain carrying visible artist pixels |
| `FIT2_PRODUCT_MESH_QUALIFIER` | exact observation-bound mesh product admission | **CONTRACT / IMPLEMENTATION CLOSED PASS** | independently rerasterizes exact promoted mesh and applies frozen coverage/topology gates; real corrected mesh result still pending fresh G/W |
| `COMPILER_MESH_SKIN_BINDING` | `QualifiedEditableMeshIR` <- exact qualified W | **BINDING** | direct S rows or legal local-convex supports only; position/raster/skin share support authority |
| `COMPONENT_MECHANICAL_ASSEMBLY` | component identity/ownership/attachments | **BINDING CONTRACT HARDENED / REAL FIT2 RESULT PENDING** | full safe-S component identity precedes view clipping; rigid labels require mechanical proof |
| `DYNAMIC_MOTION_PROOF` | qualification-owned deformation probe/proof | **BINDING MECHANICAL PROBE** | current rotation-only lane is not professional motion quality |
| `EXACT_RUNTIME_EXPORT_IDENTITY` | runtime/export consumes qualified product identities | **BINDING / RECLOSURE PENDING** | no hidden retriangulation, replacement mesh or transferred mechanics onto a second topology truth |

## Corrected FIT2 product architecture

Current same-Mage reclosure path:

`8 RGBA + exact cameras -> H1 observable evidence -> Compiler GSA/RiggingSurfaceIR S -> fresh Geppetto proposal -> Compiler QualifiedSkeletonIR G -> fresh Arachne proposal -> Compiler QualifiedSkinIR W -> directional observation-domain CDT -> FIT2 product mesh qualification -> QualifiedEditableMeshIR M -> exact mesh-skin binding B -> component/mechanical assembly -> motion -> proof -> runtime/export`

Semantic ownership rule: learned systems propose evidence or semantics; deterministic Compiler stages qualify legality, provenance, support and exact product identity. Compiler rejection/qualification may not silently substitute a second learned semantic answer.

## Directional mesh architecture — binding decision

The real E2E FIT1 contradiction proved that mesh is not an incidental debug/render helper. It is the deformation/render domain carrying artist-visible pixels.

The lower-level:

`qualify_mwb2_observation_cdt_mesh(...)`

proves legal CDT/surface-support compatibility only. It is **not** product admission.

The current FIT2 product entry point is:

`compiler/realsas_compiler_core/mesh/product_qualification.py::qualify_fit2_product_mwb2_observation_cdt_mesh(...)`

It requires exact `ObservationRasterDomain` authority and independently:

1. checks exact observation mask/source-alpha identity;
2. qualifies legal CDT/surface support;
3. reconstructs every mesh raster position from exact `SurfaceSupportBinding` coefficients and admitted S raster bindings;
4. rerasterizes the exact promoted triangles;
5. recomputes recall, precision, IoU, large-component recall and largest connected uncovered region;
6. remeasures triangle/topology quality;
7. applies frozen product thresholds fail-closed;
8. binds exact observation hashes into the resealed mesh lineage.

`candidate.residual_report` is diagnostic only and can never establish product coverage by itself.

Frozen product thresholds are preregistered in `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md` and may not be relaxed post-hoc.

Self-hosted implementation proof: workflow run `34716890157`, head `b1dc7fca6b6497d97c7be727d66fd9c0b64c3268`, runner `realsas-wsl-1660ti`, static compile PASS, `41 passed in 4.36s`.

## Supported inserted-vertex architecture

Legal inserted vertices use `LOCAL_CONVEX_INTERPOLATION`:

`P(v) = Σ a_i P(S_i)` and `W(v) = Σ a_i W(S_i)`, with `a_i >= 0`, `Σ a_i = 1`.

The same support coefficients own:

- canonical/rest geometry;
- directional raster placement;
- deterministic skin transfer.

A CDT/kernel-emitted point does not acquire product authority merely by existing. Arbitrary quality-Steiner generation remains blocked until each emitted point proves a legal admitted support simplex.

## Component identity architecture

Component identity is derived on full safe-S topology before directional visibility clipping. Unsafe/UNKNOWN relations cannot join components. A view may hide members but may not redefine canonical component identity.

Aggregate alpha coverage may not erase small but important mechanically/semantically distinct visible pieces. The current Mage witness keeps hat/cape/book/wand as explicit component witnesses after fresh G/W becomes available.

`RIGID_SKINNED_COMPONENT` requires proof against exact W. `RIGID_BONE_ATTACHMENT` requires explicit socket/bind authority. Labels alone are not mechanics evidence.

## Runtime/export identity architecture

The exact qualified product mesh is the deformation/runtime/export mesh identity.

Forbidden after product mesh qualification:

- hidden retriangulation;
- an alpha-clipped replacement mesh;
- barycentric transfer of mechanics onto a different topology;
- renderer-local topology authority;
- silently dropping qualified visible components.

Repacking, projection or sprite bake is allowed only as a projection of the exact qualified product state. Frame 0 and dynamic proof must bind the same S/G/W/M/B/component/motion lineage admitted upstream.

## Historical Arachne V5 architecture

Historical old-lineage FIT1 path:

`RiggingSurfaceIR + QualifiedSkeletonIR -> A1 V4 backbone -> K4×512 Z -> direct N×J logits -> masked row softmax -> SkinProposalIR -> Compiler QualifiedSkinIR`

Frozen checkpoint composition:

- backbone SHA-256 `95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763`;
- decoder delta SHA-256 `13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137`;
- decoder params `325,313`;
- total params `138,378,466`;
- A0 continuous-field runtime dependency `false`.

This remains useful scoped FIT1 evidence but is not current corrected Mage W authority.

## Historical evidence preserved

The following remain valid scoped evidence and must not be deleted/relabelled:

- historical Geppetto FIT1 closure;
- A0 C3/C4 sampling failures;
- A0 K4 closure/oracle;
- A1 V4 frozen-decoder failure;
- boundary/feature-preservation/pair-geometry/support-shaping diagnostics;
- 8K direct-simplex historical partial/horizon-limited result;
- 16K direct-simplex matched causal PASS;
- historical Arachne V5 FIT1 closure;
- original sparse MWB2/runtime mesh failure evidence that caused product reopening.

## Anti-conflation rules

1. FIT1 PASS != current corrected FIT2 authority.
2. FIT1 or same-Mage FIT2 != unseen-family generalization.
3. Scientific PASS != automatic promotion.
4. Historical local PASS != end-to-end product equivalence.
5. A0 representation proof != requirement to ship A0 decoder.
6. A1 V4 frozen-decoder FAIL != V4 backbone representation failure.
7. Learned proposal != Compiler canonical authority.
8. Legal CDT topology != product-valid visible deformation domain.
9. Candidate-reported coverage != independently measured product coverage.
10. Product mesh contract PASS != corrected real FIT2 mesh result PASS.
11. Runtime/export may project qualified state but may not invent a second topology truth.
12. Rotation-only probe != professional motion.
13. PRODUCT_PASS remains a separate exact end-to-end contract.
