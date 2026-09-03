# RealSaS — MWB2 Directional Behavioral Closure — 2026-09-03

**Status:** `PASS_MWB2_DIRECTIONAL_BEHAVIORAL_CLOSED`

## Boundary

This closure is behavioral authority for the deterministic RealSaS seam:

`observed S + observed local raster relations + qualified W -> MWB2 directional candidate -> QualifiedEditableMeshIR -> QualifiedMeshSkinIR -> verified LBS`

It does not claim source-mesh recovery, teacher topology recovery, hidden-surface completion, or generalization.

## Preregistered authority

Frozen behavioral gate:

- `compiler/test_mwb2_directional_behavioral_v1.py`
- `.github/workflows/mwb2_directional_behavioral_v1.yml`
- initial frozen failure run: `33780981680`
- repaired PASS run: `33782152736`

The preregistered behavioral test was not changed after observing the failure.

## Failure signature before repair

The original MWB2 producer treated every triangle clique in an eight-connected observed raster relation graph as a face. A native 2x2 lattice cell is K4 because both diagonals are relation-supported. The frozen behavioral panel measured across all clean directions:

- face-area multiplicity approximately `2.0`;
- maximum face incidence per edge `4`;
- mixed positive/negative target-view winding;
- deformation and skin transfer otherwise exact.

This isolated the defect to deterministic mesh topology generation rather than S, W, mesh-skin transfer, or LBS.

## Generic repair

`compiler/realsas_compiler_core/mwb2.py` now performs deterministic non-overlapping relation-supported triangulation in target-view raster space:

- source mesh remains forbidden;
- UNKNOWN/AMBIGUOUS/OCCLUDED/UNOBSERVED bridges remain forbidden;
- candidate faces still require a three-edge safe relation clique;
- positive-area interior overlap is forbidden;
- shared vertices and shared boundary edges remain legal;
- admitted faces are consistently CCW in the target view;
- no CDT or external numerical triangulator was promoted.

Independent source regression:

- a single K4 observed-raster cell must produce exactly two non-overlapping faces;
- relation-clique count remains four, proving the regression actually exercises the former failure;
- maximum edge incidence is at most two;
- total raster area is exactly one cell.

## Frozen panel PASS

PASS run `33782152736` executed three preregistered generic lattice/height-field witnesses over all eight directions.

### `tilted_grid_4x4`

Every direction:

- vertex coverage `1.0`;
- face count `18`;
- edge count `33`;
- Euler characteristic `1`;
- area multiplicity `1.0` within floating-point tolerance;
- max face incidence per edge `2`;
- zero degenerate faces;
- one consistent winding sign;
- deformation RMS `0.0`;
- max weight error at floating-point scale (`~1e-16`).

### `curved_grid_5x4`

Every direction:

- vertex coverage `1.0`;
- face count `24`;
- edge count `43`;
- Euler characteristic `1`;
- area multiplicity `1.0` within floating-point tolerance;
- max face incidence per edge `2`;
- zero degenerate faces;
- one consistent winding sign;
- deformation RMS `0.0`;
- max weight error at floating-point scale (`~1e-16`).

### `offset_grid_4x5`

Every direction:

- vertex coverage `1.0`;
- face count `24`;
- edge count `43`;
- Euler characteristic `1`;
- area multiplicity `1.0` within floating-point tolerance;
- max face incidence per edge `2`;
- zero degenerate faces;
- one consistent winding sign;
- deformation RMS `0.0`;
- max weight error at floating-point scale (`~1e-16`).

## Causal UNKNOWN cut

The frozen UNKNOWN-cut mutation changed a subset of safe relations to typed UNKNOWN crossings.

Observed result:

- baseline faces `18` -> mutated faces `12`;
- baseline raster area `2304` -> mutated raster area `1536`;
- crossing output faces `0`;
- source mesh used `false`.

Thus UNKNOWN relations cannot silently restore or bridge removed coverage.

## Closure verdict

`MWB2_DIRECTIONAL_BEHAVIORAL = PASS/CLOSED`

The simple deterministic producer passed its typed quality gate; **CDT promotion is not authorized or needed by this evidence**.

Next seam authority: observation-derived appearance must be tested against the actual `PIXEL_CENTER_XY` 1024 raster contract and exact per-corner provenance before architecture refreeze.
