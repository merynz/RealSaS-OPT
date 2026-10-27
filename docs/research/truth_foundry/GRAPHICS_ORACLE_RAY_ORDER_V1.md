# Graphics Oracle Ray Order V1

Status: TEACHER_ORACLE_COURT_CANDIDATE__NO_GRAPHICS_QUALIFICATION__NO_TRAINING_AUTHORITY

This court tests one consequence-bearing Graphics question before IRIS/TESSA
training: whether complete local geometric ray order on the fixed canonical mesh
can be measured without semantic layer labels and consumed by the existing fixed
Compiler Graphics logic.

## What is measured

For each declared orthographic product camera and selected pixel ray:

- every forward triangle intersection on the fixed triangulated canonical mesh;
- exact face address + barycentric address;
- forward depth;
- complete-stack status;
- depth ties under an explicit source-unit tolerance.

No back-face culling is used. Coincident/shared-edge hits are preserved as ties;
face id never resolves a tie.

## What is intentionally not measured yet

The repaired FIT8 calibration artifact declares:

- `qualified_sampler = false`;
- `linear_rgb_and_effective_alpha_measurement = NOT_RUN`.

Therefore V1 does not claim qualified effective alpha, linear RGB, source artistic
layer order, relighting, or CAA appearance closure. The fixed `graphics_v2`
consumer must report `UNKNOWN_ALPHA` for these court rays while still consuming
non-tied geometric depth into scope-local front/back constraints.

## Authority boundary

FIT8 source topology is not canonical M and must not be relabeled as canonical
mesh authority. FIT8 may later be used to test source-space ray measurement and
coverage, but the fixed-Compiler consequence court in this V1 uses the immutable
Knight `single_m_caa_knight_v1` fixture whose baseline explicitly declares
`canonical_geometry = M` and contains the eight product cameras.

The court-local admitted artifact object exists only to exercise the already
fail-closed Compiler API in source tests. It does not mint Registry qualification,
Graphics activation, ProductRevision authority, or training admission.

## Pass condition

A source PASS requires all of the following:

1. all eight Knight product views yield at least one complete multi-hit ray under
   deterministic geometry-only candidate selection;
2. measured rays produce non-empty local order constraints in `graphics_v2`;
3. ties are not resolved by face id;
4. no edge is deleted to force a total order;
5. unobserved visibility remains `UNKNOWN`;
6. alpha/RGB compositing remains explicitly invalid/unknown until renderer truth
   is separately measured and qualified.

A later behavioral court must bind these constraints into the native Knight
runtime and show a raster consequence. A later renderer adapter must separately
measure effective alpha/linear RGB/removal response before appearance can PASS.
