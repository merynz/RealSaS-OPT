# RealSaS — MWB2 Directional Behavioral Preregistration — 2026-09-03

**Status:** `PREREGISTERED_BEFORE_OUTPUT`

## Purpose

Close the deterministic product seam:

`observation-derived RiggingSurfaceIR S + Compiler-qualified W -> observed local relation complex -> build_mwb2_candidate -> qualify_mwb2_mesh -> bind_mwb2_mesh_skin -> direction-local editable mesh + mesh skin -> verified deformation consequence`.

Existing MWB0/MWB1 tests remain valid source/lineage evidence but are not product behavioral authority. MWB1 explicitly tests only a sacrificial identity-subset baseline. The existing complete synthetic E2E uses a trivial four-node square and does not establish useful directional mesh quality.

## Product obligations under test

The frozen architecture requires:

- no source mesh authority;
- no UNKNOWN/AMBIGUOUS/OCCLUDED/UNOBSERVED crossing;
- useful direction-local discretization/topology;
- surface-derived vertex bindings;
- legal skin transfer/simplex;
- actual deformation consequence consistent with bound S/W;
- exact lineage through the current Compiler mesh/skin types.

A relation graph is not itself a mesh. A valid relation complex must be converted into a non-overlapping manifold directional face complex before it can be product-authoritative.

## Synthetic observation carrier

Use an independently specified production-q-lattice-semantic carrier, not the earlier irregular capsule shell.

For every witness:

1. create a regular native anchor-pixel lattice in anchor view 0;
2. assign each anchor a finite 3D point on a generic tilted or mildly curved height field;
3. analytically reproject that exact point into eight orthographic directions;
4. emit exact `ObservationSample` rays/depths for all eight views;
5. emit `hypothesis_groups` and `hypothesis_anchor_raster` metadata with the same semantics as `observation_evidence_emitter_v2`;
6. call the actual `compile_surface_v2`, thereby exercising persistence plus `attach_observed_local_relations_v2`;
7. do not inject source mesh faces/edges or teacher topology.

The anchor lattice spacing, point field and camera projections are frozen below before output.

## Frozen witness panel

All witnesses use eight azimuthal orthographic directions at `0,45,...315` degrees around the vertical axis, a common `1024` pixel raster, vertical up `(0,1,0)`, and analytic P reconstruction from exact rays.

### `tilted_grid_4x4`

- lattice rows/cols: `4 x 4`;
- anchor stride: `16 px`;
- world lattice step: `0.05`;
- horizontal depth slope: `k = 0.41421356237309503`;
- curvature: `0`;
- center pixel: `(511.5, 511.5)`.

### `curved_grid_5x4`

- lattice rows/cols: `5 x 4`;
- anchor stride: `20 px`;
- world lattice step: `0.045`;
- horizontal depth slope: `k = 0.31`;
- quadratic curvature coefficient: `0.18` applied as `x += curvature * (u^2 - mean(u^2))`;
- center pixel: `(511.5, 511.5)`.

### `offset_grid_4x5`

- lattice rows/cols: `4 x 5`;
- anchor stride: `24 px`;
- world lattice step: `0.04`;
- horizontal depth slope: `k = -0.27`;
- curvature: `0.08` with the same centered quadratic form;
- center pixel offset: `(+37,-29)` from `(511.5,511.5)`.

No witness value may be changed after observing current MWB output.

## Exact camera/ray construction

For direction angle `theta`:

- `F = (cos(theta), 0, sin(theta))`;
- `U = (0,1,0)`;
- `R = cross(F,U)` normalized;
- screen coordinates are `sx = dot(P,R)`, `sy = dot(P,U)`;
- ray origin is `O = sx R + sy U - D F` with fixed `D=2.0`;
- ray depth is `d = D + dot(P,F)`;
- therefore `O + dF = P` exactly up to floating arithmetic.

Raster coordinates use the same fixed per-witness world-to-pixel scale selected so anchor-view world lattice step maps exactly to the preregistered anchor stride. All generated samples must remain finite and in frame.

## Frozen mechanical/skin truth for the seam

This is a deterministic MWB gate, not a Geppetto/Arachne gate.

For each compiled surface:

- create three generic control loci spanning the patch;
- emit an explicitly labeled oracle mechanical `SkeletonProposalIR` and pass it through real `Compiler.qualify_skeleton_v2`;
- define dense skin truth from normalized Gaussian distance to the Compiler-qualified control positions with fixed `sigma = 0.11`;
- emit `SkinProposalIR` and pass it through real `Compiler.qualify_skin`;
- only the resulting `QualifiedSkinIR` is admitted to MWB2.

Canonical IDs remain Compiler-owned.

## Directional mesh acceptance

Run all eight views independently through actual shipping MWB2.

For every view, PASS requires:

1. candidate/qualified mesh source-mesh flags remain false;
2. every visible admitted lattice surface node participates in the qualified mesh (`vertex_coverage = 1.0`);
3. every face has nonzero signed raster area;
4. face winding is globally consistent within the connected patch (all signed areas have one sign);
5. no undirected mesh edge belongs to more than two faces;
6. for the full rectangular observed patch, Euler characteristic `V - E + F = 1`;
7. summed absolute face raster area divided by the convex-hull raster area is within `[0.999999, 1.000001]`;
8. mesh-skin row count equals mesh vertex count;
9. every mesh-skin row has nonnegative finite weights with simplex residual `<= 1e-9`;
10. because current MWB2 vertices are identity-surface bound, transferred mesh skin must equal the corresponding qualified surface skin row to `1e-9` max absolute error;
11. verified LBS of the qualified mesh under the frozen probe transforms must match the same source-bound S/W rows to RMS `<= 1e-7`.

The area criterion measures overdraw/gaps at the directional product boundary. A graph with all four triangles of a four-corner cell is not accepted merely because every triangle is individually legal.

## Frozen deformation probes

Use four poses:

- identity;
- joint-dependent horizontal translation;
- joint-dependent vertical translation;
- mixed horizontal/depth translation.

Transform magnitudes follow the already used generic verified-LBS probe scale and are independent of witness family.

## UNKNOWN-cut causal mutation

For `tilted_grid_4x4`, after the clean baseline measurement only, create a diagnostic copy of S where every local relation crossing the central anchor-lattice vertical cut is relabeled as an explicit `UNKNOWN_BRIDGE` with `crosses_unknown=True`.

Then rerun MWB2 for view 0.

PASS requires:

- no output face may contain vertices from both sides of the cut;
- unknown relations may not increase face count or raster covered area;
- source-mesh authority remains false.

Failure of this causal mutation is a product safety failure even if the clean mesh passes.

## Panel authority

Panel PASS requires all three clean witnesses to pass all eight views plus the UNKNOWN-cut mutation.

No averaging may hide a failed view or witness.

## Causal classification

- failure before `compile_surface_v2` -> apparatus/source-contract failure;
- zero/insufficient local relations on this native lattice carrier -> observation-local-relation producer failure;
- adequate relation complex but mesh area/manifold/winding failure -> MWB2 discretization failure;
- legal mesh but skin transfer failure -> MWB2 skin-binding failure;
- legal mesh/skin but deformation mismatch -> mesh-skin consequence failure;
- UNKNOWN-cut bridge -> fail-closed topology safety failure.

## Change control

After first output, no witness geometry, lattice spacing, camera, threshold, area band, topology criterion, sigma, skin construction or deformation tolerance may be changed merely to obtain PASS.

A failure may motivate only a separately logged generic diagnostic or a generic source repair justified by the failing semantic obligation.
