# Surface Witness Sampler Court — Preregistration V1

**Status:** sealed before execution.  
**Execution class:** deterministic CPU.  
**No Product / Truth139 / FIT8 authority.**

## Problem

Current `deterministic_surface_witness_v1` selects evenly spaced vertex-array indices. Its result changes when an equivalent surface's vertex array is permuted, and it has no geometry-native coverage guarantee.

A replacement must not use semantic/component IDs as production inputs. Component labels may be used only by the court as evaluator metadata.

## Arms

All arms receive only `(xyz, normal)` and sample the same fixed witness budget.

### A — `CURRENT_INDEX_UNIFORM`

Exact current strategy: uniformly spaced array indices.

### B — `XYZ_FPS`

Deterministic farthest-point sampling in normalized xyz:

- canonical lexicographic tie-breaking on `(x,y,z,nx,ny,nz)`;
- first point = lexicographically smallest canonical row;
- iteratively choose point with largest minimum squared xyz distance to the selected set;
- ties broken by canonical order.

No component IDs, mesh IDs or source vertex indices are features.

### C/D/E — `XYZ_NORMAL_FPS_lambda`

Same deterministic FPS but distance is

`d^2 = ||delta_xyz||^2 + lambda^2 * ||delta_normal||^2`

for `lambda in {0.05, 0.10, 0.20}` after xyz is normalized by the full point-cloud bbox diagonal. Normals are unit normalized.

## Synthetic stress suite

Use fixed deterministic point sets with evaluator-only group labels:

1. `DENSE_BODY_PLUS_TINY_DISTANT_APPENDAGE`
   - dense body sphere/cloud;
   - sparse thin appendage ending in a small terminal group.

2. `NEARBY_OPPOSITE_NORMAL_SHEETS`
   - two spatially close sheets carrying opposite normals;
   - evaluates whether normal diversity is retained where xyz alone is ambiguous.

3. `LONG_THIN_CURVE_PLUS_BODY`
   - dense body plus long low-area curved support.

4. `MULTISCALE_COMPONENTS`
   - one large component and several sparse components at different separations/scales.

5. `UNIFORM_SURFACE`
   - approximately uniform sphere/cube-like surface; guards against treatments that over-prioritize rare structures and worsen global coverage.

For every stress surface, run 16 independent input-array permutations. The geometric set is identical.

## Budget

Use a fixed witness budget of 64 in V1 synthetic courts. Every stress surface has >64 points.

## Metrics

### 1. Permutation invariance

Map selected samples back to canonical `(xyz,normal)` rows. A treatment passes only if its selected canonical set is identical across all 16 input permutations for every stress surface.

### 2. Spatial cover radius

Maximum Euclidean xyz distance from any source point to its nearest selected witness, divided by the source bbox diagonal.

Report mean and worst-case across stress surfaces/permutations.

### 3. Rare-group hit recall

Evaluator-only group labels define rare/terminal/thin groups. For every such group, measure whether at least one witness lies in the group. Production sampler never receives these labels.

### 4. Normal cover

For every source point, find the nearest selected witness in xyz among the four closest selected witnesses and report angular normal mismatch; summarize p95. This prevents a spatially-close but opposite-facing surface from being treated as fully covered.

## Selection gates

`CURRENT_INDEX_UNIFORM` is retained only if it is permutation invariant and its worst-case spatial cover is within 5% of the best candidate. Otherwise it is ineligible.

A candidate FPS arm is eligible only if:

- permutation invariance = 1.0 on all stress surfaces;
- rare-group hit recall = 1.0;
- worst-case normalized spatial cover radius <= current * 0.80;
- uniform-surface cover radius <= current * 1.05.

Among eligible arms:

1. select the one with lowest worst-case spatial cover radius;
2. if within 2% relative, select lower p95 normal mismatch;
3. if still within 2%, select the simpler/lower-lambda arm in order `XYZ_FPS`, `lambda=.05`, `.10`, `.20`.

If no candidate passes, V1 makes no sampler selection.

## Interpretation limits

This tiny court establishes deterministic geometry-native behavior only. Before production replacement, the selected algorithm must be run on real extracted surfaces at the full 1,024-witness budget and compared for coverage, compute and pathological extraction pepper/noise.

A one-witness-per-connected-component rule is explicitly not an arm: extraction pepper/noise must not consume quota merely because it forms a distinct component.
