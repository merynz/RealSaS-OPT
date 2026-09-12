# RealSaS — FIT2 Mesh + Component Closure Preregistration — 2026-09-12

**Status:** `FROZEN_BEFORE_CORRECTED_FIT2_MESH_RESULT`  
**Witness:** same Mage witness, corrected full-subject FIT2 lineage  
**Product target:** automatic editable Spine-class 8-direction 2D/2.5D puppet  
**Result observed before this prereg:** **NO corrected FIT2 CDT/mesh result**

## Question

After corrected H1/GSA, Geppetto and Arachne authority is available, can RealSaS produce a directional render/deformation mesh that actually covers the observed raster subject and remains exactly bound to the admitted mechanical surface, skeleton and skin?

The old failure mode to exclude is explicit: a mesh may be topologically legal and remain inside alpha while still leaving large visible regions completely untriangulated. Aggregate face existence is not sufficient product evidence.

## Authority separation

- `RiggingSurfaceIR S` remains mechanical/geometry evidence authority.
- `QualifiedEditableMeshIR M` is the exact directional render/deformation discretization.
- `QualifiedSkinIR W` is surface-domain semantic skin authority.
- `QualifiedMeshSkinIR B` is the deterministic exact `M <- W` binding.
- Raster alpha is visible-domain coverage authority; it does not create hidden 3D geometry.
- Runtime/export may project/repack `M+B`; it may not create a second mesh truth.

A watertight 3D reconstruction is **not** required. Each view does require a sufficiently complete triangulated visible 2D deformation domain.

## Frozen mesh-quality policy

`RealSaS.MeshQualityPolicy.v1` is preregistered for corrected FIT2 product-mesh proof:

- per-view source-alpha recall `>= 0.94`;
- precision inside exact source alpha `>= 0.995`;
- alpha IoU `>= 0.935`;
- largest single uncovered 4-connected region `<= 0.015` of foreground area;
- every alpha connected component occupying at least `0.0025` of foreground must have recall `>= 0.90`;
- degenerate faces `= 0`;
- duplicate faces `= 0`;
- non-manifold edges `= 0`;
- minimum raster-space triangle angle `>= 0.25 deg`;
- maximum raster-space triangle aspect ratio `<= 250`.

These are admission thresholds, not a claim that the resulting mesh is optimally retopologized. Quality metrics remain visible in evidence even when the gate passes.

## Required V0..V7 evidence

For every direction, closure must emit from the exact qualified mesh:

1. original real observation;
2. exact mesh wireframe overlay;
3. uncovered-alpha heatmap;
4. coverage metrics including recall / precision / IoU / largest uncovered connected region;
5. mesh topology metrics including triangle count, non-manifold count, minimum angle and maximum aspect ratio;
6. manifest binding observation hash, camera hash, surface lineage hash, mesh lineage hash and renderer identity.

No synthetic image generation may substitute for these artifacts.

## Steiner / inserted-vertex rule

Inserted vertices are **allowed** and should be used when they improve directional mesh quality, but only through `LOCAL_CONVEX_INTERPOLATION` over admitted `RiggingSurfaceIR` support.

For every inserted vertex:

- support coefficients are finite, nonnegative and simplex-valued;
- rest `P` is derived exactly from the same support coefficients;
- raster rest position is derived from the same local support;
- mesh skin is the deterministic convex transfer of already-qualified surface skin rows;
- no new joint support may be invented;
- unsupported/UNKNOWN-crossing insertion fails closed.

The typed qualification/binding path for such vertices is authorized by this prereg. Automatic interior CDT quality-Steiner generation is not credited until its exact support construction is implemented and tested. Existing boundary-recovery/kernel-only points are not product authority merely because the numerical CDT emitted them.

## Component / attachment closure

`ComponentEvidenceIR -> QualifiedComponentSetIR` remains the typed component seam. FIT2 must use the stronger mechanical verification pass before product closure:

- exact surface / skeleton / skin lineage must match;
- visible surface membership may not silently disappear;
- overlapping component surface ownership is rejected;
- a `RIGID_SKINNED_COMPONENT` claim must be verified against exact `QualifiedSkinIR`, with minimum owner-joint weight `>= 0.999` and maximum non-owner mass `<= 0.001` for every member surface row;
- `RIGID_BONE_ATTACHMENT` may rely on explicit bind/socket authority but may not be inferred merely from convenient skin weights;
- detachability/swappability is never inferred from rigidity.

Mage books/wands/hat/cape remain mandatory witnesses once corrected surface membership and corrected skin are available.

## Stage order

This prereg does not interrupt the running Geppetto FIT2 experiment. Exact execution remains fail-closed:

`corrected S -> QualifiedSkeletonIR -> QualifiedSkinIR -> QualifiedEditableMeshIR -> QualifiedMeshSkinIR -> QualifiedComponentSetIR / QualifiedMechanicalAssemblyIR -> real V0..V7 deformation -> product proof -> runtime`

Mesh construction can be prepared earlier, but no mesh-skin/component product claim is made before corrected Arachne output exists.

## Stop rules

- Coverage gate fail: do not call the mesh product-ready; refine topology/support, do not relax the gate post-result.
- Large uncovered connected region fail: do not hide it behind aggregate recall.
- Missing required visible component accounting: stop.
- Rigid component mechanical evidence mismatch: stop or reclassify only through a new explicit evidence decision.
- Mesh/skin/skeleton lineage mismatch: stop.
- Runtime mesh identity drift from the exact qualified/proven mesh: stop.

## Claim boundary

Passing this gate establishes a mechanically bound, raster-covering directional product mesh on the corrected Mage witness. It does **not** establish unseen-family generalization, artist-quality motion, secondary dynamics, or global PRODUCT_PASS.
