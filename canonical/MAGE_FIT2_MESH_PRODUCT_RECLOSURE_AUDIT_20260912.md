# RealSaS — Mage FIT2 Mesh Product Reclosure Audit / Decision — 2026-09-12

**Status:** `ACTIVE__CONTRACT_AND_IMPLEMENTATION_HARDENING_IN_PROGRESS__FRESH_FIT2_RESULT_PENDING`  
**Branch:** `repair/mage-full-subject-reclosure-20260912`  
**Blocks:** `PRODUCT_PASS`, unseen/FIT8/LOFO promotion  
**Does not block:** the already-running fresh Geppetto FIT2 optimizer  
**Frozen prereg:** `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`

## Executive decision

The first real Mage end-to-end run falsified the assumption that local skeleton/skin closure implied a product-valid puppet. The product deformation domain itself is first-class authority.

For corrected FIT2, `QualifiedEditableMeshIR` is therefore not a visualization convenience. It is the exact directional domain on which artist-visible pixels are deformed and rendered. A mesh is not product-qualified merely because its triangles are legal, nondegenerate, or locally bound to `RiggingSurfaceIR`.

The current FIT2 product path is frozen as:

`corrected H1/S -> fresh QualifiedSkeletonIR G -> fresh QualifiedSkinIR W -> directional QualifiedEditableMeshIR M -> exact QualifiedMeshSkinIR B -> qualified component/mechanical assembly -> real V0..V7 deformation evidence -> motion -> exact runtime/export equivalence`

No stage may silently create a second mesh truth.

## Failure evidence that caused reopening

The real E2E forensic chain separated three different failures that had previously been conflated:

1. **Upstream subject-scope mismatch.** Historical H1 authority did not represent the complete rendered subject present in the eight RGBA observations.
2. **Mesh discretization insufficiency.** Historical conservative MWB2 relation-complex meshes covered only roughly `18%..24%` in the representative diagnostics supplied during the forensic review, while the same views had an all-safe-S convex-support ceiling around `72%..76%`. The evidence existed; the discretizer did not turn enough of it into a deformation domain.
3. **Runtime mesh identity drift.** A later demo/runtime path created an alpha-clipped barycentric mesh and transferred mechanics instead of consuming the exact qualified canonical mesh. Independent raster audit put that runtime mesh at only roughly `64%..72%` source-alpha recall.

Corrected observable H1 subsequently reached approximately `95%+` minimum alpha recall, proving that the catastrophic visible holes were not explainable as a single upstream-perception failure.

Historical exact observation-domain CDT work on the corrected substrate demonstrated approximately `90.56%..94.37%` recall with `100%` precision across the observed directional witness. That result is useful engineering evidence, but it predates fresh corrected FIT2 G/W and is **not** a current product PASS.

## What was already correctly repaired

The branch already contains substantial mesh/component hardening:

- full safe-S relation components define component identity before view clipping;
- unsafe/UNKNOWN relations may not join components;
- exact observation alpha is the admissible directional raster domain;
- historical v0.5 CDT is numerical machinery only, never source-mesh authority;
- arbitrary generated geometry is rejected fail-closed;
- `LOCAL_CONVEX_INTERPOLATION` is a typed legal support route for inserted mesh vertices when coefficients and derived geometry are exact;
- mesh-skin transfer is deterministic from admitted surface support;
- component qualification binds exact S/G/W lineage and mechanically verifies rigid-skinned ownership;
- runtime/export equivalence is required to preserve exact qualified mesh identity.

The preregistration freezes the product-mesh thresholds before a fresh corrected FIT2 mesh result exists.

## Integration gap found in this audit

The branch had a subtle but important split between **policy** and **admission**.

`RealSaS.MeshQualityPolicy.v1` already froze strict product thresholds, and `observation_domain.py` already measured recall, precision, IoU, connected foreground recall, and largest uncovered connected region. However the low-level `qualify_mwb2_observation_cdt_mesh(...)` compatibility gate still admitted candidates with a historical coarse recall floor of `0.90` and precision floor of `0.995`.

That lower-level gate is useful as a typed CDT/surface-lineage compatibility gate, but it is insufficient as product proof. In particular it does not by itself enforce:

- product recall `>= 0.94`;
- alpha IoU `>= 0.935`;
- largest uncovered 4-connected region `<= 0.015`;
- large-alpha-component recall `>= 0.90`;
- zero degenerate/duplicate/non-manifold faces;
- raster minimum-angle / maximum-aspect gates.

This is exactly the class of integration gap that allowed locally legal artifacts to be mistaken for product closure during FIT1.

## Decision / implementation

A distinct fail-closed FIT2 product admission seam now exists:

`compiler/realsas_compiler_core/mesh/product_qualification.py`

Authoritative entry point:

`qualify_fit2_product_mwb2_observation_cdt_mesh(...)`

It performs, in order:

1. low-level current-authority CDT/surface-support qualification;
2. exact raster-space topology remeasurement from the qualified mesh and admitted S bindings;
3. evaluation against frozen `FIT2_PRODUCT_MESH_QUALITY_POLICY_V1`;
4. fail-closed rejection on any product coverage/topology invariant;
5. re-sealing of the mesh lineage with explicit product-qualification provenance.

The lower-level `qualify_mwb2_observation_cdt_mesh(...)` is therefore **not** product closure evidence. It remains a compatibility/substrate gate only.

Regression coverage is added in:

`tests/compiler/test_fit2_product_mesh_qualification_v1.py`

The critical regression explicitly demonstrates that a candidate with `0.92` source-alpha recall can satisfy the historical `0.90` CDT floor while being rejected by the FIT2 product qualifier. A second regression forbids hiding a large connected hole behind high aggregate recall.

Implementation commits in this audit transaction begin at:

- `3ff6c9ed2da8fd3045f4313b90f4ea6ce000c60d` — strict FIT2 product mesh qualifier;
- `76d654cc3b47a75cf1338dbc571df9c25ee0380a` — regression tests for product admission.

## Frozen product mesh policy

Per `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`:

- source-alpha recall `>= 0.94`;
- precision inside exact source alpha `>= 0.995`;
- alpha IoU `>= 0.935`;
- largest uncovered 4-connected region `<= 0.015` of foreground;
- every alpha connected component occupying at least `0.0025` of foreground has recall `>= 0.90`;
- degenerate faces `= 0`;
- duplicate faces `= 0`;
- non-manifold edges `= 0`;
- minimum raster-space triangle angle `>= 0.25 deg`;
- maximum raster-space triangle aspect ratio `<= 250`.

These thresholds may not be relaxed after observing the corrected FIT2 result without a new explicit preregistration and fresh result lineage.

## Supported-Steiner decision

Inserted vertices are allowed only when they remain compiler-owned consequences of admitted S evidence.

Legal route:

`P(v) = Σ a_i P(S_i)`, with `a_i >= 0`, `Σ a_i = 1`, and the same coefficients governing raster placement and deterministic skin transfer.

This is `LOCAL_CONVEX_INTERPOLATION` authority, not hallucinated geometry. Automatic interior quality-Steiner insertion remains uncredited until every emitted point can construct and prove that support relation. Boundary-recovery/kernel-internal points do not become product authority merely because CDT emitted them.

## Component / attachment decision

Aggregate alpha recall may not hide loss of semantically or mechanically distinct visible pieces. Mage hat/cape/book/wand remain required witnesses once fresh corrected G/W is available.

Component identity is derived from full safe-S topology before directional visibility clipping. Product closure must prove exact S/G/W membership/ownership; visible members may not silently disappear. `RIGID_SKINNED_COMPONENT` requires mechanical verification against exact W, while `RIGID_BONE_ATTACHMENT` requires explicit bind/socket authority.

## Runtime identity invariant

The exact product mesh proven above is the mesh consumed by deformation/runtime/export.

Forbidden after product qualification:

- hidden retriangulation;
- alpha-clipped replacement mesh generation;
- barycentric mechanics transfer onto a different runtime mesh;
- a renderer-local second topology truth;
- dropping qualified components because they are inconvenient to package.

Frame 0 and dynamic-frame proof must bind the same qualified skeleton, skin, mesh, mesh-skin, component and motion lineage hashes that were admitted upstream.

## Current closure state

| Layer | State | Meaning |
| --- | --- | --- |
| corrected H1 observable authority | CLOSED | current full-subject observable geometry evidence |
| GSA8192 | CLOSED | current corrected S substrate |
| fresh Geppetto FIT2 | RUNNING | no PASS claimed yet |
| fresh Arachne FIT2 | BLOCKED ON GEPPETTO | no corrected W yet |
| mesh quality policy/prereg | FROZEN | thresholds sealed before result |
| mesh/component implementation hardening | IMPLEMENTED | typed support, component identity, strict product qualifier exist |
| corrected FIT2 directional mesh result | PENDING G/W | no product mesh PASS claimed |
| exact mesh-skin/component assembly | PENDING | requires fresh corrected W/M |
| real V0..V7 deformation proof | PENDING | must consume exact qualified identities |
| professional motion | OPEN | rotation-only probe is not product motion |
| exact runtime/export reclosure | OPEN | no second mesh truth allowed |
| PRODUCT_PASS | OPEN / BLOCKED | may not be claimed yet |

## Execution after Geppetto closes

Do not change the running Geppetto gate or optimizer because of this mesh audit.

When Geppetto closes:

1. seal fresh QualifiedSkeletonIR + V0..V7 evidence;
2. run fresh Arachne on corrected S + new G;
3. seal fresh QualifiedSkinIR;
4. build directional CDT candidates from exact observation authority;
5. admit them only through the FIT2 product mesh qualifier;
6. bind exact mesh skin from W through the same support coefficients;
7. qualify components/mechanical assembly;
8. emit real V0..V7 source / wireframe / uncovered-heatmap / deformation evidence;
9. run dynamic motion proof;
10. prove exact runtime/export lineage equivalence.

Only that chain can close the reopened Mage product witness.

## Claim boundary

This audit closes an implementation/authority hole; it does **not** claim a corrected FIT2 mesh result.

Strongest allowed claim now:

`The corrected FIT2 pipeline has a frozen, fail-closed product-mesh qualification seam that prevents the old legal-but-visibly-incomplete mesh class from being promoted. Fresh Mage product mesh closure remains pending fresh Geppetto/Arachne mechanics and real V0..V7 evidence.`
