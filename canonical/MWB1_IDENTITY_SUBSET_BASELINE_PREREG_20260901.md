# RealSaS — MWB-1 Identity / Subset Deterministic Baseline Prereg — 2026-09-01

**Status:** `FROZEN_BEFORE_MWB1_OUTPUTS__NO_NUMERICAL_SOLVER_PROMOTION__NO_LEARNED_OPTIMIZER`

## Purpose

Execute the first real-witness test of the typed mesh / mesh-weight seam added by MWB-0.

MWB-1 is deliberately not a product-quality mesh experiment. Its sole question is whether a mesh assembled from already admitted surface nodes can travel through the exact current typed lineage, weight-binding, product-state and proof invalidation route without creating a second geometry or skin authority.

## Real witness authority

Use the already sealed consumer-interlock fixture reconstructed by:

`experiments/consumer_interlock_20260829/run_exact_compiler_consumer_interlock_v0.py`

Required fixture transport SHA-256:

`81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5`

The witness provides:

- 512 admitted `RiggingSurfaceIR` nodes from the clean D2 fixture;
- sacrificial G0 proposals routed through the real Compiler skeleton qualifier/global graph optimizer;
- Compiler-minted canonical `J:*` IDs;
- A0 skin proposal routed through the real Compiler skin qualifier.

No new corpus asset, teacher identity, hidden full mesh or learned output may be introduced.

## Mesh baseline

Construct one deterministic local sacrificial triangle using only existing `RiggingSurfaceIR.surface_nodes`.

Selection rule, frozen before outputs:

1. sort surface nodes lexicographically by `surface_id`;
2. for each anchor in that order, collect other nodes that share at least one support view with the anchor;
3. sort candidates by Euclidean distance to the anchor then `surface_id`;
4. inspect the nearest 64 candidates;
5. select the first pair, in deterministic nested order, for which all three nodes share at least one support view and the triangle cross-product magnitude exceeds `1e-12 * bbox_diag^2`;
6. if no triangle exists, MWB-1 fails closed.

This selection is a topology preflight only. It does not claim complete surface coverage or product triangulation quality.

Every mesh vertex uses:

`SurfaceSupportBinding(mode="IDENTITY_SURFACE_NODE", coefficients=((surface_id,1.0),))`

and its rest position must exactly equal the admitted surface-node position.

The mesh contains exactly one face and its three boundary edges.

## Weight baseline

For every selected mesh vertex, copy the exact corresponding `QualifiedSkinIR` row onto the canonical mesh vertex.

Requirements:

- same canonical joint IDs;
- same influence ordering;
- same floating-point weight values;
- no new joint support;
- no renormalization in the binder;
- transfer method recorded as `IDENTITY_SURFACE_NODE_EXACT_COPY_V1`.

## V2 product-state test

Assemble:

`S + G + W + M + B -> CanonicalPuppetGraph.v2`

The product must bind exact hashes for:

- surface;
- skeleton;
- skin;
- mesh;
- mesh-skin.

## Mutation / stale-proof tests

### Topology mutation

Reverse the selected face vertex order while retaining the same admitted vertices and support bindings.

Required:

- mesh lineage hash changes;
- V2 product-state hash changes;
- a proof bound to the original product is rejected as stale by the mutated product.

This tests product-state identity, not geometric superiority of one winding.

### Weight mutation

Choose the first copied mesh row, in canonical mesh-vertex order, that has at least two positive influences. Transfer `delta = min(1e-6, 0.25 * min(w0,w1))` from the first positive influence to the second, preserving finite/nonnegative/simplex validity and the existing joint support set.

Required:

- mesh-skin lineage hash changes;
- V2 product-state hash changes;
- the original proof is rejected as stale.

If no selected row has at least two positive influences, MWB-1 fails closed rather than changing the rule post hoc.

## PASS criteria

`PASS_MWB1_IDENTITY_SUBSET_BASELINE` requires all of:

1. real fixture transport hash exact;
2. 512 admitted surface nodes reconstructed;
3. real Compiler skeleton qualification succeeds;
4. real Compiler skin qualification succeeds for all 512 rows;
5. deterministic local identity triangle is found;
6. every mesh rest position is exact admitted surface position;
7. every mesh skin row is an exact copy of its source qualified-skin row;
8. V2 product assembly passes all MWB-0 validators;
9. topology mutation changes mesh/product hashes and rejects stale proof;
10. valid weight mutation changes mesh-skin/product hashes and rejects stale proof;
11. scientific optimizer steps remain 0.

## Claim boundary

Permitted after PASS:

> The current typed V2 product route can bind a real admitted identity-subset mesh and exact qualified skin rows with correct lineage and stale-proof invalidation.

Forbidden:

- product-quality mesh generation is solved;
- face coverage across the full observed character is solved;
- CDT is required or qualified;
- Arachne field transfer quality is solved;
- deformation non-inferiority is established;
- hidden-surface completion is needed or allowed.

## Next gate

Only after MWB-1 PASS may MWB-2 consider local convex inserted vertices and/or exact-predicate CDT promotion under a separately frozen candidate policy.
