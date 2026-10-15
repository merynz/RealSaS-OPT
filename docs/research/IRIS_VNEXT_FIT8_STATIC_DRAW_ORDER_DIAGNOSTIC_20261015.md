# IRIS vNext FIT8 Static Draw-Order Diagnostic — 2026-10-15

## Authority state

**Status:** `DIAGNOSTIC_LOCAL_REPLAY__NOT_CANONICAL_STATIC_RAY_ARTIFACT__NOT_FIT8_PASS`

This document records a real-source diagnostic that was used to decide which presentation-capacity hypotheses deserve further court time. It is **not** a scientific FIT8 pass and it does **not** freeze the replacement draw-order mechanism.

The evidence chain has two different authority levels and they must not be conflated:

1. **Exact-source / importer / triangulation evidence** is bound to selected FIT8 bytes, a measured pinned-uFBX build receipt, and executable RealSaS qualification contracts.
2. **Static ray sweep values** below were produced by a local replay of the formulas in `tools/corpus/fit8_static_ray_census_v1.py`. They have not yet been emitted as canonical repository/platform artifacts by that producer. They are therefore diagnostic evidence only.

The executable decision logic that consumes future canonical ray summaries is `tools/corpus/fit8_static_draw_order_capacity_court_v1.py`.

## Exact selected FBX source identity

| Subject | Exact selected source SHA-256 | Bytes |
|---|---|---:|
| Snake | `553092a2e1c68ab4e49ce5c139942228a77a911b25c78635aca7901db0a8d53d` | 422252 |
| T-Rex | `db4ae6f657f32ce2253851c3eeca70602be2572b7f5e7fdda3c4c6e7132dea8c` | 1074060 |
| Spider | `01e2289dcc20cb1636b8339be79b9981990bc7e3bb6bda066aa7824caab33946` | 1233484 |
| Shark | `fee1cfe578643cefb06457332b5d323f4883eba2515eb255ae0f2abb497a75c9` | 101788 |

These hashes match `canonical/FIT8_DIVERSITY_SELECTION_V1.json`.

## Pinned uFBX build evidence

Workflow run: `38753568064` (`iris-vnext-ufbx-probe-build-v1`) — **PASS**.

Measured build receipt:

- expected and observed uFBX revision: `5955c5c0b042ac2dc6f32955ab206aaec0620cfc`
- `ufbx.c` SHA-256: `4a956c26a708e40d82ecb0aeaee54da709d5ba7c144c9642174970059bda7e1d`
- `ufbx.h` SHA-256: `f787be529af577efc04f3e3e735844a08556718dbb613d1211c2b313d72895dc`
- v3 probe source SHA-256: `fe58392d09a1ad180ee1b0a95e44e352e99a1e41d88a844c8bd2fdc2e0b49e7d`
- compiled probe binary SHA-256: `fd0cb06ef8cd493484c5edff14f3f8d0c3c26052931f9d4b098d4acfa73b368c`
- compiler SHA-256: `1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26`
- canonical build-receipt identity SHA-256: `cf6b6c90846d1a8cf7507358c0fd54ee668ae9765f5c0e77217aeb1a9f22f75b`

The v3 probe resolves FBX face material identity through the source face-material slot and node-instance material table. Its triangulation is still proposal evidence until the independent RealSaS polygon-triangulation court admits it.

## Real FBX triangulation qualification

The pinned v3 probe was run on all four exact selected FBX byte streams. The resulting proposed triangulations were checked using `RealSaS.PolygonTriangulationQualification.v1`.

All four had zero:

- triangle-count mismatch polygons,
- triangle vertex-scope violation polygons,
- polygon-boundary incidence violations,
- internal-diagonal incidence violations,
- internal diagonals aliasing authored boundary edges,
- internal diagonals reused across polygons.

Qualified static source topology from those proposals:

| Subject | Source polygons | Triangles | Components | Boundary edges | Non-manifold edges | Surface class |
|---|---:|---:|---:|---:|---:|---|
| Snake | 822 | 1618 | 3 | 0 | 0 | all closed |
| T-Rex | 838 | 1820 | 1 | 0 | 0 | closed |
| Spider | 1380 | 2712 | 15 | 0 | 0 | all closed |
| Shark | 328 | 644 | 1 | 0 | 0 | closed |

This result strongly supports the hypothesis that the previously observed T-Rex non-manifold signal came from the older triangulation/probe semantics. It does not by itself prove the complete historical causal chain; a direct old-vs-new replay would be required for that stronger claim.

## Static all-hit diagnostic — N=128

The following values are from the local replay and are therefore diagnostic only.

### Shark

- component count: 1
- source material regions: 2
- diagnostic dominant-mechanical-joint owner regions: 8
- maximum distinct component stack: 1 in all views
- p95 distinct component stack: 1 in all views
- mixed component-pair count: 0 in all views

### T-Rex

- component count: 1
- source material regions: 5
- diagnostic dominant-mechanical-joint owner regions: 26
- maximum distinct component stack: 1 in all views
- p95 distinct component stack: 1 in all views
- mixed component-pair count: 0 in all views

Absence of a mixed pair on these subjects is **not** evidence that a whole-component painter order is sufficient generically. It only means this subject/view court did not falsify it.

### Snake

- component count: 3
- source material regions: 7
- diagnostic dominant-mechanical-joint owner regions: 12
- max distinct component stack by view: `[2,2,3,2,2,2,3,2]`
- p95 distinct component stack by view: `[1,1,1,1,1,1,1,1]`
- mixed unordered component-pair counts by view at N=128: `[2,1,1,1,0,1,1,1]`
- ordered component-pair relation count by view: `[4,3,4,3,2,3,4,3]`

A single global total order on the three whole components cannot satisfy a pair that appears in both front-to-back orientations within one view.

### Spider

- component count: 15
- source material regions: 2
- diagnostic dominant-mechanical-joint owner regions: 34
- max distinct component stack by view: `[6,5,5,5,6,5,5,5]`
- p95 distinct component stack by view: `[4,3,3,3,4,3,3,3]`
- mixed unordered component-pair counts by view at N=128: `[12,9,6,4,0,4,6,9]`
- ordered component-pair relation count by view: `[48,50,43,45,36,45,43,50]`

Spider is the strongest current static counterexample to whole-component painter ordering.

## Resolution stability diagnostic

### Snake

| Resolution | Mixed-pair counts by view | Max-stack by view | p95-stack by view |
|---:|---|---|---|
| 64 | `[2,0,0,1,0,1,0,0]` | `[2,2,3,2,2,2,3,2]` | `[1,1,1,1,1,1,1,1]` |
| 128 | `[2,1,1,1,0,1,1,1]` | `[2,2,3,2,2,2,3,2]` | `[1,1,1,1,1,1,1,1]` |
| 256 | `[2,1,1,1,0,1,1,1]` | `[2,2,3,2,2,2,3,2]` | `[1,1,1,1,1,1,1,1]` |

R128 and R256 have the same mixed-pair view pattern and the same ordered-pair counts. The maximum stack is invariant across all three resolutions.

### Spider

| Resolution | Mixed-pair counts by view | Max-stack by view | p95-stack by view |
|---:|---|---|---|
| 64 | `[10,9,6,3,0,3,6,9]` | `[6,5,5,5,6,5,5,5]` | `[4,3,3,3,4,3,3,3]` |
| 128 | `[12,9,6,4,0,4,6,9]` | `[6,5,5,5,6,5,5,5]` | `[4,3,3,3,4,3,3,3]` |
| 256 | `[14,9,7,4,0,4,7,9]` | `[6,5,5,5,6,5,5,5]` | `[4,3,3,3,4,3,3,3]` |

The exact mixed-pair count changes modestly with sampling resolution, but the existence of mixed ordering, maximum stack, and p95 stack remain stable. This makes a pure raster-alias explanation implausible for the observed capacity failure.

## Decision

### Falsified candidate

**Whole geometric component → one painter slot → one per-view total order** is **falsified as a generic presentation primitive** by the current real-source static diagnostic. Snake and Spider contain component pairs whose relative order reverses across image-space support in the same canonical view.

The executable court encodes this asymmetrically:

- mixed pair observed → `COMPONENT_TOTAL_ORDER_FALSIFIED`;
- no mixed pair observed → `COMPONENT_TOTAL_ORDER_NOT_FALSIFIED`, never `PASS`.

### What is not decided

This diagnostic does **not** choose among:

1. per-pixel / geometric depth resolution,
2. qualified subcomponent fragment ordering,
3. view/pose-conditioned fragment DAGs,
4. hybrids of the above with CAO ownership/decomposition.

It also does not establish that camera-Z or a z-buffer alone is sufficient for RealSaS presentation semantics. Static geometric visibility and authored 2D presentation ownership remain distinct courts.

### Required closure before architecture freeze

1. Produce canonical `RealSaS.FIT8StaticRayCensus.v1` artifacts from the repository producer for the exact subjects, not a local mirror replay.
2. Run `RealSaS.FIT8StaticDrawOrderCapacityCourt.v1` on those artifacts.
3. Extend the same evidence model to selected authored animation frames, because a static court cannot determine pose-conditioned ordering capacity.
4. Measure fragment granularity only after dynamic overlap evidence exists; do not freeze a fragment DAG by intuition.
5. Keep diagnostic mechanical-owner IDs separate from generic semantic/appearance ownership authority.

## Claims explicitly not made

- no FIT8 pass,
- no product pass,
- no full-truth bundle closure,
- no dynamic draw-order closure,
- no semantic-owner authority from dominant skin joints,
- no final CAO fragment granularity decision,
- no final choice between z-buffer, fragment ordering, or DAG presentation.
