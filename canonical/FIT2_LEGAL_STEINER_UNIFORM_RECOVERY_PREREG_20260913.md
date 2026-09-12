# FIT2 Legal Steiner Uniform Recovery Preregistration — 2026-09-13

Status: `PREREGISTERED_BEFORE_RECOVERY_SWEEP`

## Question

The legal-Steiner ceiling diagnostic closed `BOUNDARY_RECOVERY_GEOMETRICALLY_SUFFICIENT` on all eight Mage FIT2 views. The next question is whether a **real emitted, surface-supported mesh** can close the frozen product mesh quality policy without new geometry authority.

## Frozen treatment family

For each current post-contraction supported CDT kernel triangle, perform a conforming uniform barycentric subdivision using one global integer edge subdivision factor `n` from the ordered set:

`[2, 3, 4]`

All kernel triangles in the view are subdivided at the selected factor. Shared edge vertices are canonicalized by exact rational support coefficients, so adjacent parent triangles share the same generated vertices and no T-junction authority is introduced.

Every generated vertex MUST be bound to admitted `S` using:

- `IDENTITY_SURFACE_NODE` when the coefficient vector is one-hot; otherwise
- `LOCAL_CONVEX_INTERPOLATION` with nonnegative coefficients summing to one.

Rest `P`, raster XY, and later skin weights are derived from the same support coefficients. No free XY/world vertex is admitted.

After subdivision, a subtriangle is admitted only when the exact `ObservationRasterDomain.triangle_inside()` predicate passes. No triangle may spill outside exact source alpha.

## Forbidden

- cross-component bridges;
- unknown-relation bridges;
- teacher/source mesh topology;
- arbitrary unconstrained Steiner points;
- threshold changes;
- per-view hand tuning;
- post-result treatment-set expansion;
- lineage relabeling.

## Selection rule

Run `n=2`, then `n=3`, then `n=4`. For each `n`, evaluate all eight views with the existing frozen `FIT2_PRODUCT_MESH_QUALITY_POLICY_V1`.

The selected treatment is the **first global n** for which all eight views pass the complete frozen policy, including:

- source-alpha recall >= 0.94;
- precision >= 0.995;
- alpha IoU >= 0.935;
- largest uncovered component <= 0.015 foreground;
- large alpha component recall >= 0.90;
- zero degenerate faces;
- zero duplicate faces;
- zero nonmanifold edges;
- min raster triangle angle >= 0.25 deg;
- max raster triangle aspect ratio <= 250.

If no preregistered `n` passes all eight views, the treatment closes FAIL. Do not add levels after seeing the result.

## Authority / replay prerequisite

The exact product GSA lineage remains `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da` and MUST NOT be relabelled.

Current Colab numeric replay can reconstruct a byte-different GSA lineage. Therefore this recovery sweep is allowed to continue in **diagnostic scope only** when all of the following hold:

1. node count = 8171;
2. edge count = 23656;
3. observed/completed = 7391/780;
4. support counts = `[2537,2765,2194,3021,2772,2871,2183,2803]`;
5. the historical sealed CDT baseline structural+raster metrics replay exactly on V0..V7.

A recovery mesh may be called `FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC` under that scope, but **not PRODUCT PASS** until the exact sealed GSA authority is materialized/consumed or exact lineage replay is restored.

## Outputs

For every treatment/view record:

- vertices / faces / edges;
- generated local-convex vertex count;
- recall / precision / IoU;
- largest uncovered component;
- min triangle angle / max aspect ratio;
- degenerate / duplicate / nonmanifold counts;
- exact frozen-policy failure invariants;
- mesh-only textured render.

Emit a treatment summary and select only by the frozen rule above.
