# FIT2 Baseline-Preserving Adaptive Patch CDT — Preregistration

**Date:** 2026-09-13  
**ID:** `FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_PREREG_20260913`  
**Status:** `PREREGISTERED_BEFORE_EXECUTION`

## Scientific question

Can the exact sealed FIT2 observation-domain CDT replay remain the **coverage authority**
while the previously tested adaptive support-derived Delaunay solver is restricted to
**local quality repair and residual recovery**, producing one final supported mesh that
passes the unchanged frozen FIT2 product-mesh policy?

This experiment is motivated by two already-observed facts:

1. Uniform global legal-Steiner subdivision `n={2,3,4}` is falsified as a final family:
   it preserves bad parent shape and does not reliably recover V2/V6 coverage.
2. Global adaptive reconstruction `B4_G16 -> B2_G12 -> B1_G8` is falsified as a final
   family: accepted triangles are legal and quality-clean, but global Delaunay followed
   by locality/alpha rejection destroys coverage. `B1_G8` retained 100% alpha precision
   and frozen angle/aspect compliance while recall remained only about 70%.

The new test therefore does **not** ask either family to rebuild the character globally.

## Frozen authority split

- **Coverage authority:** exact sealed baseline CDT replay, V0..V7.
- **Quality-repair operator:** local adaptive support-derived CDT only.
- **Residual-recovery operator:** local CDT only inside a supported parent triangle
  that was absent from the original baseline due observation-alpha admission.
- **Final output:** one mesh per view. There is no overlapping "baseline mesh + second
  render mesh" product representation.

The baseline must reproduce the sealed historical/current structural+raster metrics for
all eight views before any treatment is allowed to execute.

## Frozen treatment order

Treatments are tried in this exact order and selection stops at the first **global
eight-view full frozen-policy PASS**:

1. `P1_B2_G10`
   - repair neighborhood: 1 baseline face-adjacency hop
   - adaptive alpha/support boundary stride: 2 px
   - adaptive interior grid spacing: 10 px
2. `P2_B1_G8`
   - repair neighborhood: 2 hops
   - boundary stride: 1 px
   - interior spacing: 8 px
3. `P2_B1_G6`
   - repair neighborhood: 2 hops
   - boundary stride: 1 px
   - interior spacing: 6 px

No treatment may be added, removed, reordered, or changed after observing this run.

## Stage A — KEEP / REPAIR

The exact baseline candidate is the starting state.

A baseline face is **KEEP** when it already satisfies the frozen raster triangle quality
limits. A face failing minimum angle or maximum aspect is a **REPAIR seed**.

REPAIR seeds are expanded only by the treatment's preregistered baseline face-adjacency
hop count. Connected expanded sets form local cavities.

For each admissible disk-like cavity:

- the cavity's existing baseline boundary cycle is a fixed constrained seam;
- existing good geometry outside the cavity is untouched;
- adaptive raster samples are taken only inside the cavity's old raster support;
- every adaptive sample must map to an admitted supported parent triangle;
- every generated point must be exactly `IDENTITY_SURFACE_NODE` or
  `LOCAL_CONVEX_INTERPOLATION` over admitted S;
- unbound quality-Steiner generation is disabled;
- every replacement triangle must satisfy the frozen angle/aspect limits;
- every replacement triangle must remain inside both the old cavity raster domain and
  the exact observation alpha;
- every original seam edge must survive;
- duplicate, degenerate, and non-manifold topology is forbidden.

If any condition fails, that cavity is rolled back to the exact baseline state.

## Stage B — RECOVER

Residual recovery is allowed only in a supported kernel parent triangle that was
**absent from the original exact baseline**.

For each such parent:

- the original three admitted S carriers are the constrained local boundary;
- adaptive samples are taken only from currently uncovered exact-alpha pixels within
  that parent;
- all generated points use the same support-simplex rule as REPAIR;
- unbound quality-Steiner generation remains disabled;
- only triangles fully admitted by exact alpha and frozen angle/aspect policy survive;
- cross-parent, cross-component, UNKNOWN-relation and extrapolative bridges are
  forbidden.

The parent patch is accepted only if exact raster coverage strictly increases and all
monotonic/topology conditions below hold. Otherwise it is rolled back.

## Monotonic acceptance law

Every accepted patch, whether REPAIR or RECOVER, must satisfy:

```text
old_covered_pixels ⊆ new_covered_pixels
new_covered_pixels ⊆ exact_observation_alpha
precision_new >= precision_old
duplicate_faces = 0
degenerate_faces = 0
nonmanifold_edges = 0
support_legality = PASS
```

REPAIR additionally must preserve its complete old cavity raster support. RECOVER must
strictly add exact-alpha coverage.

The experiment may never trade away already-proven baseline coverage to improve triangle
shape.

## Frozen final mesh policy

No thresholds are changed:

- source alpha recall >= 0.94
- precision inside alpha >= 0.995
- alpha IoU >= 0.935
- largest uncovered 4-connected region <= 0.015 of foreground
- every alpha component >= 0.0025 foreground has recall >= 0.90
- degenerate faces = 0
- duplicate faces = 0
- non-manifold edges = 0
- minimum raster triangle angle >= 0.25 degrees
- maximum raster triangle aspect ratio <= 250

Selection is the first treatment that passes this policy on **all eight views**.

## Forbidden

- source/teacher mesh topology
- source-mesh vertices as geometry authority
- cross-component bridges
- UNKNOWN/unsafe relation bridges
- negative support coefficients
- extrapolation outside a support simplex
- post-hoc threshold relaxation
- post-hoc treatment changes
- global adaptive reconstruction
- global uniform Steiner subdivision
- relabelling the frozen GSA authority lineage

## GSA replay scope

The expected frozen GSA lineage remains
`65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`.

If the current numeric runtime reconstructs a different byte-exact GSA lineage but the
sealed CDT baseline replay closes exactly, this experiment may report only:

`FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC__NOT_PRODUCT_PASS`

It may not claim canonical PRODUCT PASS and may not rewrite the frozen GSA lineage.

## Failure interpretation

If all three preregistered treatments fail, stop. Do not add density, hops, or a fourth
treatment from observed results. The result must instead identify whether the remaining
blocker is:

- unrepaired boundary/hull quality,
- residual alpha/support geometry,
- local cavity topology,
- or another frozen mesh-policy invariant.
