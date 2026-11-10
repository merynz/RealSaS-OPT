# Corpus370 — Source raster projection and bind-rest repair pilots (2026-11-03)

Scope: **source-backed bounded engineering diagnostics, NOT 370 admission or training authority**. PR [#327](https://github.com/merynz/RealSaS-OPT/pull/327) remains draft; historical failures are immutable.

## Proven independent input separation

- 370 candidates = 294 original prepared/computed slots plus 76 preparation failures. Of those 76, exact original reasons divide into **47 mask** and **29 bind-vertex**. New `tools/corpus/classify_corpus370_observation_failures_v1.py` constructs the hash-bound per-subject failure worklist; it never silently changes FAIL to MASKED/PASS. Actual 26 additional compute execution failures remain open separately.
- Source-linked native evidence was exported read-only in Actions run 41725086471 (six subjects / 48 full render views), 41739083858 (Swordfish/Sunfish/Lionfish/Cube Pets monkey), and 41740642074 (remaining 10 Quaternius fish). Each slice bound the original bootstrap, raw source, imported geometry, cameras and original rasters.
- **FBX Dragon:** run 41740129975 compiled the independent `tools/corpus/ufbx_bind_scene_probe_v2.c`, checked original FBX SHA and Blender imported.json SHA, and verified full *per-index, per-object* geometry parity. The unskinned `Eyes` is parented below the `Face` source bone. Body 554 vertices has relative error 9.388940105308493e-8, Eyes 124 vertices after source-authored bind frame has relative error 2.41738632585657e-7 relative to full source bounding scale; unchanged original bound 1e-5. Historical V1 probe unchanged. This is numeric parity only, not camera/raster admission.
- **glTF:** independently sourced Mako and Rogue bind-rest accessories closed the raw-reader discrepancy to 2.1126216318821817e-7 and 1.9845313426212707e-7 relative errors. V2 reader leaves skinned bodies alone; ambiguous bone/skin bindings fail closed. Their legacy artifacts remain unchanged.

## Source raster mask: root cause and negative controls

The historical SAT pixel/triangle projection used an **absolute** 1e-12 px² degenerate-area guard. A nearly collinear projected triangle can extend 101 pixels yet have only ~1e-5 px² numerical signed area from source/import float32 rounding. Pixel-square SAT then paints a long phantom one-pixel line, while the original native Blender foreground alpha has no corresponding coverage.

Versioned `iris_observation_admission_v3.py` now treats the projected orientation as uncertain when its determinant falls within a conservative float32 coordinate-error interval. The derivation and exact formula are in `canonical/CORPUS370_SOURCE_PROJECTION_PRECISION_PILOT_20261103.json`. This removes **only unobservable projected raster support**, not the 3D triangle, independent geometry/UDF or the source image. The **original 2px discrepancy threshold is not modified**. Other unmatched reference pixels are recorded as UNKNOWN, never labeled numeric-zero negatives. Historical v1/v2 code stays intact.

On **13 original fish failure views**, the original mask gaps for Swordfish 101px, Sunfish 76px, GoblinShark 61px, Betta 68px, Humphead 68px, RedSnapper 60px, Turbot 66px, FlowerHorn 41px, CoralGrouper 25px, BlackLionFish 34px and Tetra 24px became at most √2px under the unchanged gate. Lionfish and CardinalFish remain **2.236px FAIL**. This is **11 repaired selected failed-view projections, NOT 11 admitted fish subjects**: their other six/seven authentic views and new source/Attempt bindings are still pending. The 48 previously sampled real views retained *every* original foreground raster pixel; none became alpha-positive/reference-negative under the new projection.

Kenney Cube Pets and other mask-failure packs require separate class-specific investigation rather than adopting the fish explanation universally. The original 47 are not yet closed.

## Strict next dependencies

1. Run the V4 observation preparation/revalidation with exact pinned Blender, the V3 numerical admission and **V2 native FBX executable**; keep eight actual views and original import JSON bytes in the new packet. No silent reuse of historical V1/V2 data.
2. Independently validate source/observation parity on all 370 and resolve the remaining 47 mask/29 bind-vertex failures with typed outcomes, preserving source/transport errors as execution failures.
3. Issue a new Platform EngineRelease/child Attempts for the unreleased Corpus370 V3 family graph (Stage50 exact surface independent of Stage51 calibration) and a representative real-data end-to-end pilot.
4. Integrate the standalone source-relative TESSA sampled-contact proof as a **new independent consumer**, do not relax legacy V5; finish model-facing canonical projection, local vs global sign scope, and true loss masks.
5. Only after that scope closes, issue the **370 repair/revalidation Run All notebook**. Materialization and training are distinct subsequent gates.

This checkpoint makes no eligibility, unseen split, materialization or training claim.
