# Corpus370 — Variable-K original source sample compositor and original K2 regression

2026-11-06 · PR #351 · source-graphics research only. **No artist owner or model training authority**.

## Why

Previous original Triceratops physical palette full-film V1/V2 tests rejected apparently collateral pixels. Their old "source-ray center/cardinal" and 3x3 finite source stencil criteria do not establish the actual Cycles pixel footprint: original source triangle geometry has diagonal material boundaries crossing a pixel interior. V1 FAIL (1 apparent collateral) and V2 FAIL (4) are preserved unchanged.

V18m/V18n **already solved exact matching Cycles sample-index G/R depth ordering**, and original UV/material source replay under source-private original 3D geometry; no need to invent another segmentation feature. New tools/corpus/compose_corpus370_source_sample_k_v1.py generalizes the same-source sample mechanism to variable K >= 1, with original physically isolated source RGBA, independent same-sample source View-Z per layer, geometry/material-alpha separation, premultiplied Porter-Duff OVER in local nearest-source-depth order, unknown depth ties within .0002, and fail-closed invalid source depth.

## Immutable real-source regression

GitHub Actions [42245131250](https://github.com/merynz/RealSaS-OPT/actions/runs/42245131250): **2/2 jobs PASS** (synthetic K1/K2/K3 contracts + original real source K2 regression court). Original immutable source Triceratops V18n raw material cycle samples from 42137097177-1; V18m original geometry depth from 42134142993-1; source SHA ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a. SHA sealed original source material array SHA 1431ac67fb4e98b7f53ccf0e22f8d035b214a865498dfccd32a5a17f295ea0f8, source depth array SHA ece272a9ebb3aabe5350e7d877dbc4a13047f6416c569444b104423345f84eca.

Input to the new source K=2 predictor is **ONLY physically isolated G/R source material RGBA and same-index source depth**. Original SOURCE_NATIVE and SOUP_FULL never select depth or compositing order; they are court targets. Four source cameras V0/V2/V4/V6 x 8 sample offsets (0,4,8,12,16,20,24,28) x 192x192 source-pixel grid = **1,179,648 source sample positions**. All source samples qualified with zero UNKNOWN/ties and 0 original source native/soup RGBA sample failures at preregistered .025, zero mismatches versus original independently-courted V18n K2 predictor. The prior V18n raw status literal (known typo) was left immutable, not "fixed" here.

[Private original K2 independent variable-K regression receipt](https://drive.google.com/file/d/1ag62YzMBdse9upmfinQZnMUdU_svWcNI/view).

This is an independent **compatibility regression**, not new generic K3 source proof or artist drawable discovery. No arbitrary alpha/material transparency / hidden-source reconstruction or IRIS inference is qualified.

## V3 science gate

[Preregistered original 3-palette same-Cycles-sample experiment](CORPUS370_ORIGINAL_SOURCE_VARIABLE_K3_CYCLES_PALETTE_EDIT_PREREG_20261106.md) tests true source original RGB palette color-equivalence operations (3 groups, NOT artist part count) with physically isolated original triangles, their Cycles source-depth, independently recolored isolated color layers and source FULL/EDITED_FULL counterfactual evaluation targets. Original rest source; four directions V1/V3/V5/V7 (previously seen by a different film sampler); 8 identical 1-spp source subset indices; 13 render arms; source-private 416 render experiment. Synthetic tests are PASS; actual K3 source physical court must be checked separately.

Two first pre-science source-run blockers were recorded separately: original FBX raw polygons are not guaranteed triangles, so source isolate must come from exact evaluated source loop-triangle canonical order; and Blender 4.0 original opaque alpha policy must be replayed explicitly in Blender4.5 using *existing V18n SourceAppearancePolicyReplay*, not re-inferred from current importer blend mode. Neither blocker is a visual model failure, neither gives ownership authority.

Scientific next target: independent original K3 sample-indexed physical edit consequence court; if PASS, carry typed per-source, per-view, per-sample visibility/compositing facts to a **candidate Graphics Evidence IR** while ownership remains UNKNOWN; Cat texture/UV and true multi-pose edited source still separate outstanding truth families. Do not elevate any source palette group into unconditional SAME_DRAWABLE or DIFFERENT_DRAWABLE labels.
