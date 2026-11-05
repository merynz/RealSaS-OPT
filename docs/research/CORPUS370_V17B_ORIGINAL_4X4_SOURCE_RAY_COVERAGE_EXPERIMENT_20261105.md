# Corpus370 V17b — frozen 4×4 physical subpixel source coverage falsifier

Date: 2026-11-05. Branch `corpus370-v17b-subpixel-source-ray-coverage-v1`, draft stacked PR #341 on independently measured V17 PR #339, ultimately V16 real SHA-courted native original pose controls. **Do not claim PASS on real original sources until private court results are read back.**

## Why subpixels are a scientific necessity

In V17's original Cat/Triceratops/Cowfish pilot, the real V16 source joint mesh was queried at a deterministic geometry-centroid-derived **pixel centre**. 1,547/1,728 rays had a complete, untied, nonempty source triangle hit stack; **181 failed**, including 132 *actually changed* premultiplied RGBA pixel instances. A mesh triangle may cross a raster pixel but not its centre, so blank centre ray and nonzero anti-aliased Cycles RGBA are not necessarily inconsistent. Equally, an error in camera projection, source scene, raster convention or geometric source alpha can cause a mismatch. Before attributing pixel response to anonymous mechanical/graphics contributors, independently measure these hypotheses.

V17b pre-registers the exact **4×4 grid** (fractional pixel x/y offsets `[-0.375,-0.125,0.125,0.375]`) and runs **16 full source face ray intersections** per *original V17 unknown ray* (never recolor- or label-conditioned), at the exact original V14 camera and original V16 animated + signed joint source vertices. The fixed source image pixel coordinate, V14/V16 source SHA, V17 original court and original RGB/alpha observation are unchanged. The stage computes the count of complete, untied subpixel rays and labels a previously unknown centre ray only as `SOME_GEOMETRIC_SUBPIXEL_SOURCE_HITS__ALPHA_AND_ARTIST_OWNER_UNKNOWN` when at least one physically valid subray is found.

The separate independent V17b court reopens original V17 private per-pixel proposal, original 3D geometry/camera, V16 joint vertex arrays and V14/V16 pixel evidence; recomputes every 4×4 ray query and independently reconstructs each valid subpixel source-world barycentric point. It compares the entire proposal with the remeasured result and rejects tampering/forged art labels. A null result (no recoverable subray) remains a **valid scientific falsification**, not proof there is no artist layer.

**Nonpromotion:** 4×4 source geometry coverage != exact stochastic Cycles pixel filter/alpha; a source triangle hit does not imply opaque source material, compositor visibility, human-editable drawable grouping, same/different owner, or product draw-order authority. The original V12 heldout source group edits proved that convenient K partitions can fail on new targets. Thus V17b is only the projection/raster precondition for a subsequent genuine source editability and alpha/peel court, not an art semantics replacement.

## Execution and custody

- Producer: `tools/corpus/measure_corpus370_v17b_original_subpixel_coverage_v1.py`.
- Independent court: `tools/corpus/court_corpus370_v17b_subpixel_coverage_v1.py`.
- Adversarial tests: `tests/corpus/test_corpus370_v17b_subpixel_coverage_v1.py`.
- Exact private source host: `tools/corpus/hosted_corpus370_v17b_source_subpixel_coverage.sh` (original V5 hydration for source camera and rest faces; read only frozen V14/V16/V17 source evidence, **no Blender rerender**).
- Workflow: `.github/workflows/corpus370-v17b-real-original-4x4-source-subpixel-geometry.yml`, GitHub-hosted 3-source matrix, SHA-readback to private Drive.
- Original source art pixels, per-ray original source triangle addresses and barycentric details are NOT committed to Git. Public aggregate may only be committed after all original source jobs and independent SHA courts are verified.

Training and Materialization Run All must remain blocked on independent ownership/alpha/appearance truth, regardless of the numerical 4×4 ray recovery.
