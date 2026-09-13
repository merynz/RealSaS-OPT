# FIT2 Adaptive Boundary + Quality Delaunay Recovery — Preregistration — 2026-09-13

Status: `PREREGISTERED__BEFORE_FIT2_REAL_RUN`

Branch authority: `repair/mage-full-subject-reclosure-20260912-adaptive-steiner-tmp`

## 1. Scientific question

The uniform legal-Steiner family `n={2,3,4}` is falsified: increasing global barycentric subdivision improves coverage but preserves the shape of bad parent triangles, so the frozen angle/aspect failures persist. At the same time the legal-Steiner ceiling diagnostic proves that the existing supported CDT kernel union contains enough observation-domain area to clear the frozen coverage thresholds in all eight views.

The next experiment asks whether a **support-local adaptive remesh** of the exact supported kernel/alpha intersection can emit a real mesh that clears the complete frozen FIT2 mesh policy without cross-component bridges, unknown-relation bridges, source-mesh topology, teacher topology, or threshold changes.

## 2. Frozen authority and prerequisites

The run MUST:

1. verify the corrected H1 zero-surface and all V0..V7 observation/camera SHA-256 values already sealed by the Mage FIT2 reclosure;
2. rebuild the current GSA substrate and preserve the canonical expected GSA lineage `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da` as authority;
3. if byte-exact GSA replay differs under the current numeric runtime, NEVER relabel it PASS; diagnostic continuation is allowed only after cardinality/support invariants match and the sealed historical CDT baseline V0..V7 structural+raster metrics replay exactly;
4. replay the historical sealed CDT baseline before any adaptive treatment.

A successful emitted-mesh treatment while GSA byte-exact lineage remains open is therefore only `FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC__NOT_PRODUCT_PASS`.

## 3. Geometry authority

The adaptive remesh may consume only:

- current admitted `RiggingSurfaceIR S`;
- current safe full-S component partition;
- current supported pre-alpha CDT kernel triangle union;
- exact directional observation alpha;
- deterministic derived raster sample points.

It MUST NOT consume source/teacher mesh topology.

Every generated mesh vertex MUST be derived from one containing supported kernel triangle and carry either:

- `IDENTITY_SURFACE_NODE`, or
- `LOCAL_CONVEX_INTERPOLATION`

with finite nonnegative coefficients summing to one. Rest `P` and cached raster position MUST be re-derived from those exact coefficients.

No triangle may connect different full-safe surface components.

## 4. Adaptive point treatments

Exactly three treatments are preregistered, in this order:

1. `B4_G16`: boundary cell stride 4 px, interior grid spacing 16 px;
2. `B2_G12`: boundary cell stride 2 px, interior grid spacing 12 px;
3. `B1_G8`: boundary cell stride 1 px, interior grid spacing 8 px.

Selection rule: **first treatment that passes the full frozen mesh policy in all eight views**.

No extra treatment may be appended after seeing FIT2 results. A new family requires a new preregistration.

Boundary samples are selected from the exact pixelwise target `observation_alpha AND supported_component_kernel_union`. Interior samples are deterministic regular-grid samples from the same target. Every sample must be mapped into a containing original supported kernel triangle; unmappable samples are recorded and excluded, never guessed.

## 5. Support-local Delaunay rule

For each full-safe component independently:

1. construct the deterministic sample cloud;
2. map each point to one original supported kernel triangle by deterministic barycentric containment;
3. run SciPy Delaunay on that component's cloud;
4. accept a Delaunay triangle only when all conditions hold:
   - nondegenerate;
   - all three sample parents are topologically local within two parent-triangle adjacency hops;
   - exact rasterized triangle pixels lie within the component's supported kernel-union mask;
   - `ObservationRasterDomain.triangle_inside()` passes against exact alpha;
   - the frozen per-face raster quality floors already hold: min angle >= 0.25 degrees and aspect <= 250.

Rejected triangles are diagnostic evidence. There is no bridge or extrapolation fallback.

## 6. Frozen product mesh policy

Thresholds are unchanged:

- source alpha recall >= 0.94;
- precision >= 0.995;
- alpha IoU >= 0.935;
- largest uncovered 4-connected region <= 0.015 foreground;
- each alpha component >= 0.0025 foreground must have recall >= 0.90;
- degenerate faces = 0;
- duplicate faces = 0;
- non-manifold edges = 0;
- min raster triangle angle >= 0.25 degrees;
- max raster triangle aspect <= 250.

The final qualified mesh is remeasured from its `SurfaceSupportBinding` against exact observation authority. Candidate-reported coverage is not sufficient evidence.

## 7. Required artifacts

For every attempted treatment and view record:

- sample counts and unmappable counts;
- Delaunay simplex counts;
- rejection counts by locality/support/alpha/quality/degeneracy;
- emitted vertex/edge/face counts;
- local-convex vertex count;
- full frozen policy metrics and failure invariants;
- candidate and qualified mesh lineage hashes.

If a global treatment passes, render real mesh-only textured evidence for V0..V7 plus a contact sheet.

If none passes, still write the complete failure manifest and render the final preregistered treatment for diagnosis. No product PASS may be claimed.

## 8. Decision contract

Allowed terminal decisions:

- `PASS__FIRST_GLOBAL_FULL_FROZEN_MESH_POLICY__<TREATMENT>`
- `FAIL__NO_PREREGISTERED_ADAPTIVE_BOUNDARY_QUALITY_TREATMENT_PASSES_ALL_VIEWS`

Uniform `n={2,3,4}` remains falsified regardless of this experiment's outcome.
