# IRIS vNext FIT8 Static Draw-Order — Exact Repository Execution — 2026-10-15

## Authority state

`EXACT_REPO_PRODUCER_LOCAL_EXECUTION__STATIC_GEOMETRY_VISIBILITY_ONLY__NOT_PLATFORM_ARTIFACT__NOT_FIT8_PASS`

This document upgrades the earlier local-mirror diagnostic with byte-verified repository producer execution. It still does **not** claim FIT8 pass, product pass, dynamic presentation closure, or a persisted platform research artifact.

## Exact producer source snapshot

The static-ray producer source was packaged by GitHub Actions only after its regression suite passed, then downloaded and executed locally against the exact selected source-derived FBX probe evidence.

Source snapshot workflow run: `38756149381` — **PASS**.

Snapshot commit:

- `8daf29511af4b4bd1c25210606fabc6081aacb1e`

Verified Git blob identities:

- `fit8_static_ray_census_v1.py`: `e788553e12e3a996471f328595163aebc5fbfbaf`
- `fit8_static_ray_census_v2.py`: `875b13f1e0f385f06d37c3394928c33e0332b602`
- `fit8_static_draw_order_capacity_court_v1.py`: `d41a2fb04675371466d63d0415473d6760d660d0`
- `fit8_polygon_triangulation_qualification_v1.py`: `4165e785b6424e82f7bd5af5df08318baddda693`
- `fit8_surface_topology_invariants_v1.py`: `390ca6654dc17fbb9abd95c221263b2eef6bf1e7`
- `fit8_rendered_mechanical_census_contract_v1.py`: `85b3a34f49d53b57ff473b249a07c379f0b5227f`

The local files were re-hashed as Git blobs before execution and matched these branch identities.

## Owner-authority correction

`RealSaS.FIT8StaticRayCensus.v1` inherited a dangerous metadata shortcut: if any nonnegative `face_owner_id` existed, its summary wrote `mechanical_owner_authority=true`.

That is invalid for the current FBX carriers because their owner labels are explicitly:

`DIAGNOSTIC_DOMINANT_NORMALIZED_RAW_SKIN_JOINT`

They are not generic semantic-owner truth and not qualified mechanical-owner authority.

`RealSaS.FIT8StaticRayCensus.v2` was therefore introduced. It reuses the same tested v1 geometry, camera, raster and all-hit implementation, but separates:

- owner labels present,
- owner authority state,
- diagnostic-only owner labels,
- qualified owner authority.

For all four real FBX executions below:

- `mechanical_owner_authority_state = DIAGNOSTIC`
- `mechanical_owner_authority = false`
- `mechanical_owner_diagnostic_only = true`
- `owner_authority_inferred_from_label_presence = false`

The v2 regression court passed before the exact-source execution.

## Exact source and importer lineage

Selected FBX source SHA-256 values:

| Subject | Source SHA-256 |
|---|---|
| Snake | `553092a2e1c68ab4e49ce5c139942228a77a911b25c78635aca7901db0a8d53d` |
| T-Rex | `db4ae6f657f32ce2253851c3eeca70602be2572b7f5e7fdda3c4c6e7132dea8c` |
| Spider | `01e2289dcc20cb1636b8339be79b9981990bc7e3bb6bda066aa7824caab33946` |
| Shark | `fee1cfe578643cefb06457332b5d323f4883eba2515eb255ae0f2abb497a75c9` |

Pinned importer build evidence:

- observed uFBX revision: `5955c5c0b042ac2dc6f32955ab206aaec0620cfc`
- v3 probe binary SHA-256: `fd0cb06ef8cd493484c5edff14f3f8d0c3c26052931f9d4b098d4acfa73b368c`
- canonical build-receipt identity SHA-256: `cf6b6c90846d1a8cf7507358c0fd54ee668ae9765f5c0e77217aeb1a9f22f75b`

Every FBX triangulation was re-qualified using the exact repository `PolygonTriangulationQualification.v1` bytes before the v2 static ray execution.

## N=128 exact-repo results

| Subject | Components | Owner state | Mixed-pair views | Mixed-pair count sum | Maximum distinct component stack | Component total-order state |
|---|---:|---|---:|---:|---:|---|
| Snake | 3 | DIAGNOSTIC | 7 / 8 | 8 | 3 | `COMPONENT_TOTAL_ORDER_FALSIFIED` |
| T-Rex | 1 | DIAGNOSTIC | 0 / 8 | 0 | 1 | `COMPONENT_TOTAL_ORDER_NOT_FALSIFIED` |
| Spider | 15 | DIAGNOSTIC | 7 / 8 | 50 | 6 | `COMPONENT_TOTAL_ORDER_FALSIFIED` |
| Shark | 1 | DIAGNOSTIC | 0 / 8 | 0 | 1 | `COMPONENT_TOTAL_ORDER_NOT_FALSIFIED` |

Cohort court state:

`COMPONENT_TOTAL_ORDER_FALSIFIED`

Cohort canonical court hash:

`efde6d853cfc1b44004c5e332658cd8c3eca005d2613af4c61ed36626df1abaa`

The same capacity result was obtained before and after the v2 owner-authority correction, as expected: draw-order capacity is based on geometric component ordering and does not consume owner authority.

## Mixed-relation structure

### Snake

Union of mixed unordered component pairs across the eight canonical views:

- `(0, 1)`
- `(0, 2)`

No mixed relation exists between components 1 and 2 in this execution. Component 0 is the only mixed-order hub.

At N=128 the mixed-pair counts by view are:

`[2,1,1,1,0,1,1,1]`

### Spider

The union of mixed unordered component pairs is:

`(0,k)` for every `k ∈ {1,...,14}`.

Thus component 0 has mixed-order degree **14 / 14** relative to all other geometric components. The mixed relation is not scattered among arbitrary pairs: one central component repeatedly moves from in-front-of to behind every other component depending on image-space location and view.

At N=128 the mixed-pair counts by view are:

`[12,9,6,4,0,4,6,9]`

This is a strong structural explanation for why one painter slot per whole component is not a generic RealSaS presentation primitive.

## Resolution stability with exact repository producer

### Snake

- N=64: mixed views `3/8`, mixed-pair sum `4`, maximum stack `3`
- N=128: mixed views `7/8`, mixed-pair sum `8`, maximum stack `3`
- N=256: mixed views `7/8`, mixed-pair sum `8`, maximum stack `3`

The N=128 and N=256 capacity result is identical.

### Spider

- N=64: mixed views `7/8`, mixed-pair sum `46`, maximum stack `6`
- N=128: mixed views `7/8`, mixed-pair sum `50`, maximum stack `6`
- N=256: mixed views `7/8`, mixed-pair sum `54`, maximum stack `6`

The exact number of sampled mixed relations changes slightly with resolution, but the capacity failure and maximum stack remain stable.

## Architecture decision supported by this court

### Rejected

`whole geometric component -> one painter slot -> one per-view total order`

is rejected as a **generic** static presentation primitive.

The rejection is now supported by:

1. exact selected source bytes,
2. measured pinned uFBX build provenance,
3. exact repository triangulation qualification,
4. exact repository static-ray v2 producer bytes,
5. exact repository draw-order capacity court bytes,
6. resolution-stable real morphology counterexamples.

### Not selected yet

The evidence does **not** select a replacement. Still-open candidates include:

- per-pixel / geometric depth resolution,
- qualified subcomponent fragmentation,
- view/pose-conditioned fragment ordering or DAGs,
- a hybrid where geometry handles ordinary visibility and CAO overrides qualified presentation ownership.

### Important limitation still open

The current capacity court is triggered by explicit mixed A↔B relations. A whole-component total order can also fail through longer directed cycles such as `A→B→C→A` even when no pair is individually mixed. The next court revision must therefore consume the complete ordered relation graph, not only mixed pairs.

## No overclaim

This execution is stronger than the earlier local-mirror diagnostic, but it is still not:

- a platform-persisted Attempt artifact,
- a full eight-subject rendered/mechanical census,
- a dynamic draw-order court,
- a CAO visible-coherence pass,
- a FIT8 pass,
- a product pass.
