# RealSaS — FIT2 Legal Steiner Ceiling Diagnostic Preregistration — 2026-09-13

**Status:** `FROZEN_BEFORE_LEGAL_STEINER_CEILING_RESULT`  
**Branch:** `repair/mage-full-subject-reclosure-20260912`  
**Witness:** corrected Mage FIT2 / sealed GSA8192  
**Purpose:** decide whether support-certified boundary recovery can close the existing frozen product-mesh raster gates without changing S authority or relaxing thresholds.

## Observed baseline that motivates this diagnostic

The exact historical `f02` CDT geometry was replayed on the corrected GSA8192 substrate and matched all geometry-affecting structural / raster evidence for V0..V7. The resulting diagnostic mesh is not product-qualified. It shows:

- minimum recall approximately `0.90557`;
- precision `1.0` in all eight views;
- largest uncovered connected region approximately `0.02617` of foreground;
- several views already near or above the frozen `0.94` recall floor, while V2/V6 remain the strongest coverage failures.

The existing frozen product mesh policy remains unchanged. This prereg does not reinterpret or weaken any admission threshold.

## Scientific question

Given the exact same admitted `RiggingSurfaceIR` carriers, safe-component partition, historical/current CDT cells, and exact observation alpha domain, is the missing raster area geometrically reachable by **support-certified `LOCAL_CONVEX_INTERPOLATION` boundary recovery**?

If the answer is no, do not spend time implementing blind Steiner refinement. Return to substrate/support/component evidence.

If the answer is yes, implementation of a bounded, fail-closed recovery pass is authorized by the already-frozen inserted-vertex rule in `FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`.

## Ceiling definition

For one view, reproduce the same post-boundary-contraction CDT kernel triangles used by the current adapter **before** whole-triangle alpha rejection.

Every retained kernel triangle must have three exact admitted S carriers as vertices. No generated/unsupported kernel vertex may survive into this set.

For each such triangle `T = (S_a, S_b, S_c)`, any point inside `T` has a constructive convex support:

`P = a P(S_a) + b P(S_b) + c P(S_c)`

with `a,b,c >= 0` and `a+b+c = 1`. The same coefficients can define the directional raster position and, once fresh W exists, deterministic skin transfer.

The **legal Steiner pixel ceiling** is therefore:

`exact observation alpha ∩ raster-union(post-contraction supported CDT kernel triangles)`.

This is a **pixelwise support ceiling**, not an emitted mesh and not a product PASS. It asks whether arbitrary sufficiently fine support-certified boundary subdivision could in principle cover those alpha pixels while maintaining zero outside-alpha spill.

## Forbidden shortcuts

The ceiling must not use:

- source/teacher mesh topology;
- cross-component bridges;
- UNKNOWN/unsafe relations to define component identity;
- arbitrary convex hulls unrelated to the actual supported CDT cells;
- generated kernel vertices without an exact S support construction;
- alpha dilation/erosion or threshold changes;
- post-hoc changes to the frozen product policy.

## Required exact checks

Before computing a ceiling for each view, the diagnostic must reproduce the existing baseline candidate and verify its geometry-affecting values against the sealed historical CDT report:

- vertex count;
- edge count;
- face count;
- kernel triangle count;
- alpha-rejected face count;
- constraint split count;
- contracted boundary-recovery vertex count;
- post-contraction triangle count;
- foreground / predicted pixel counts;
- recall and precision.

Any drift is fail-closed.

Historical candidate/mesh lineage hashes remain outside this diagnostic gate because the historical report omitted the exact `camera_binding_hash` preimage; that provenance defect is recorded separately and does not change geometry authority.

## Decision rule

Evaluate the ceiling only against the **coverage subset** of the already-frozen `RealSaS.MeshQualityPolicy.v1`:

- recall `>= 0.94`;
- precision `>= 0.995`;
- IoU `>= 0.935`;
- largest uncovered 4-connected region `<= 0.015` of foreground;
- every alpha connected component occupying at least `0.0025` of foreground has recall `>= 0.90`.

The ceiling does **not** evaluate triangle angle/aspect/non-manifold/duplicate/degenerate constraints because it is not an emitted topology.

### Outcome A — `BOUNDARY_RECOVERY_GEOMETRICALLY_SUFFICIENT`

All eight ceiling masks satisfy the frozen coverage subset. Proceed to implement a bounded recovery pass inside the existing `MeshDiscretizationCandidateIR` authority:

- inserted vertices only through exact `LOCAL_CONVEX_INTERPOLATION`;
- same coefficients for P, raster position and later W transfer;
- same full-safe component only;
- exact alpha containment;
- monotonic coverage improvement;
- full product mesh quality gate still required afterward.

### Outcome B — `BOUNDARY_RECOVERY_GEOMETRICALLY_INSUFFICIENT`

At least one view fails the frozen coverage subset even at the pixelwise support ceiling. Do not implement blind Steiner refinement. Diagnose the residual unreachable regions as a support/substrate/component problem.

## Required evidence

For V0..V7 emit:

1. real source observation;
2. baseline **mesh-only textured render** (source texels shown only where the actual current CDT triangles rasterize);
3. ceiling mesh-only textured render;
4. recoverable-only mask;
5. unreachable-only mask;
6. baseline vs ceiling recall / IoU / largest-hole / component-recall table;
7. exact baseline replay assertions and ceiling decision manifest.

Wireframe remains optional debug evidence; it is not the main product-facing visualization.

## Claim boundary

A PASS here means only:

`The exact current supported CDT cells contain enough observation-domain area that a support-certified local-convex boundary recovery implementation is geometrically capable of satisfying the frozen raster-coverage subset.`

It does not prove an emitted legal Steiner mesh, mesh quality, fresh W binding, component closure, deformation quality, runtime equivalence, or PRODUCT_PASS.
