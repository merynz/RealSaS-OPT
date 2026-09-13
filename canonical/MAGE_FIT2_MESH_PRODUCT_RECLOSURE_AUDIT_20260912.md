# RealSaS — Mage FIT2 Mesh Product Reclosure Audit / Decision — 2026-09-12

**Status:** `CLOSED_PASS__CONTRACT_AND_IMPLEMENTATION_HARDENING_ONLY__FRESH_FIT2_MESH_RESULT_PENDING`  
**Branch:** `repair/mage-full-subject-reclosure-20260912`  
**Blocks:** `PRODUCT_PASS`, unseen/FIT8/LOFO promotion until fresh product result closes  
**Does not block:** the already-running fresh Geppetto FIT2 optimizer  
**Frozen prereg:** `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`  
**Self-hosted contract evidence:** workflow run `34716890157`, `realsas-wsl-1660ti`, `41 passed in 4.36s`

## Executive decision

The first real Mage end-to-end run falsified the assumption that local skeleton/skin closure implied a product-valid puppet. The product deformation domain itself is first-class authority.

For corrected FIT2, `QualifiedEditableMeshIR` is therefore not a visualization convenience. It is the exact directional domain on which artist-visible pixels are deformed and rendered. A mesh is not product-qualified merely because its triangles are legal, nondegenerate, locally bound to `RiggingSurfaceIR`, or accompanied by optimistic candidate-reported coverage metrics.

The current FIT2 product path is frozen as:

`corrected H1/S -> fresh QualifiedSkeletonIR G -> fresh QualifiedSkinIR W -> directional QualifiedEditableMeshIR M -> exact QualifiedMeshSkinIR B -> qualified component/mechanical assembly -> real V0..V7 deformation evidence -> motion -> exact runtime/export equivalence`

No stage may silently create a second mesh truth.

## Failure evidence that caused reopening

The real E2E forensic chain separated three failures that had previously been conflated:

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

## Integration gaps found and closed in this audit

The branch had a subtle but important split between **policy** and **admission**.

`RealSaS.MeshQualityPolicy.v1` already froze strict product thresholds, and `observation_domain.py` already measured recall, precision, IoU, connected foreground recall, and largest uncovered connected region. However the low-level `qualify_mwb2_observation_cdt_mesh(...)` compatibility gate still admitted candidates with a historical coarse recall floor of `0.90` and precision floor of `0.995`.

That lower-level gate is useful as a typed CDT/surface-lineage compatibility gate, but it is insufficient as product proof. It does not by itself enforce the complete frozen FIT2 product policy.

A second integrity issue was found while closing the first gap: a product qualifier must not simply trust `candidate.residual_report` for coverage, even when that candidate is lineage-sealed. Product coverage is evidence against the exact observation authority and must be independently recomputed from the exact mesh that is being promoted.

Both gaps are now closed.

## Decision / implementation

The fail-closed FIT2 product admission seam is:

`compiler/realsas_compiler_core/mesh/product_qualification.py`

Authoritative entry point:

`qualify_fit2_product_mwb2_observation_cdt_mesh(...)`

It requires the exact `ObservationRasterDomain` and performs, in order:

1. validate the observation-domain view, mask hash, source-alpha hash and dimensions against the candidate's exact-alpha authority;
2. run low-level current-authority CDT/surface-support qualification;
3. reconstruct every qualified mesh vertex's raster location from its exact `SurfaceSupportBinding` and the admitted `RiggingSurfaceIR` raster bindings;
4. independently rerasterize the exact qualified mesh against the exact observation mask;
5. recompute recall, precision, IoU, connected foreground recall and largest uncovered connected region — candidate-reported coverage is diagnostic only;
6. independently remeasure raster-space mesh topology/quality;
7. evaluate the frozen `FIT2_PRODUCT_MESH_QUALITY_POLICY_V1` fail-closed;
8. re-seal mesh lineage with explicit product-observation hashes and qualification provenance.

The lower-level `qualify_mwb2_observation_cdt_mesh(...)` is therefore **not** product closure evidence. It remains a compatibility/substrate gate only.

Regression coverage lives in:

`tests/compiler/test_fit2_product_mesh_qualification_v1.py`

The critical regressions prove that:

- a sealed candidate can report apparently excellent coverage and still be rejected when the exact promoted mesh has a large missing region;
- the historical `0.90` lower-level floor is not adopted as the product threshold;
- candidate-reported coverage cannot override exact rerasterization;
- a large connected hole fails even when reported aggregate metrics look excellent;
- exact observation-mask hash drift fails closed.

## Implementation / verification lineage

Relevant commits:

- `9dd533eb5c48408faa92bcfb3469b7d32fd39368` — earlier mesh coverage/component contract hardening;
- `ebe7358dbdafbf65068dde91f3eba39b58b784b6` — earlier extended self-hosted contract tests;
- `3ff6c9ed2da8fd3045f4313b90f4ea6ce000c60d` — initial strict FIT2 product mesh admission seam;
- `76d654cc3b47a75cf1338dbc571df9c25ee0380a` — initial product-admission regressions;
- `c4c8edf959750a004544dec1063074fe0943602c` — product qualifier + tests wired into the self-hosted reclosure workflow;
- `3e8ef20b5b6ddbbec0a2b1c57912c7ccc21d30d8` — exact observation-authority rerasterization added;
- `b1dc7fca6b6497d97c7be727d66fd9c0b64c3268` — regressions updated for exact rerasterization and hash drift.

The intermediate run at `3e8ef20...` failed only because it intentionally changed the qualifier signature before the follow-up test update landed: `36 passed / 3 TypeError` from the old call sites. This is preserved as transaction history, not hidden.

Final self-hosted evidence for this hardening transaction:

- workflow: `mage-full-subject-reclosure-contract`;
- run id: `34716890157`;
- head: `b1dc7fca6b6497d97c7be727d66fd9c0b64c3268`;
- runner: `realsas-wsl-1660ti`;
- static compile: PASS;
- reclosure contract suite: `41 passed in 4.36s`;
- conclusion: `success`.

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

Frame 0 and dynamic-frame proof must bind the same qualified skeleton, skin, mesh, mesh-skin, component and motion lineage hashes admitted upstream.

## Current closure state

| Layer | State | Meaning |
| --- | --- | --- |
| corrected H1 observable authority | CLOSED | current full-subject observable geometry evidence |
| GSA8192 | CLOSED | current corrected S substrate |
| fresh Geppetto FIT2 | RUNNING | no PASS claimed yet |
| fresh Arachne FIT2 | BLOCKED ON GEPPETTO | no corrected W yet |
| mesh quality policy/prereg | FROZEN | thresholds sealed before result |
| mesh/component product-qualification hardening | CLOSED PASS | implementation + self-hosted contract proof closed |
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
5. admit them only through the FIT2 product mesh qualifier with the exact `ObservationRasterDomain`;
6. bind exact mesh skin from W through the same support coefficients;
7. qualify components/mechanical assembly;
8. emit real V0..V7 source / wireframe / uncovered-heatmap / deformation evidence;
9. run dynamic motion proof;
10. prove exact runtime/export lineage equivalence.

Only that chain can close the reopened Mage product witness.

## Claim boundary

This audit closes the **mesh product qualification contract/implementation** hole; it does **not** claim a corrected FIT2 mesh result.

Strongest allowed claim now:

`The corrected FIT2 pipeline has a frozen, self-hosted-tested, fail-closed product-mesh qualification seam that independently rerasterizes the exact promoted mesh against exact observation authority, so neither a lower-level legal CDT PASS nor optimistic candidate-reported coverage can masquerade as product closure. Fresh Mage product mesh closure remains pending fresh Geppetto/Arachne mechanics and real V0..V7 evidence.`
