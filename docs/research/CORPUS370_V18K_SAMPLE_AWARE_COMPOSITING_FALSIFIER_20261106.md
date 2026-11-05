# Corpus370 V18k — geometry-precommitted sample-aware compositing canary

Date: 2026-11-06 (Türkiye local). **PREREGISTERED DESIGN FOLLOWED BY ORIGINAL-SOURCE 3/3 INDEPENDENT COURT PASS; REPORT BELOW; NO MODEL/DRAWABLE AUTHORITY.** Draft PR #343, branch `corpus370-v18b-reactive-stain-source-edit-court-v1`. Do not equate any prospective PASS with drawable identity, original art alpha or model training authority.

## Exact research claim to falsify

V18j discovered that original Triceratops V18i RGB-only compositor failures frequently co-occur with opposing nearest G/R sides across 4×4 subpixel ray samples (21/25 retrospectively chosen failing pixels, 0/60 pass pixels). It was **outcome-selected** and was **not** a blind test. To avoid repeating that bias, V18k selects pixels using **only exact original posed source geometry and fixed V18a candidate groups** before any V18h RGBA is consulted. A joint ray/sample-composited opaque two-color image is compared with:
  1. post-rasterization `G OVER R` and `R OVER G` of independently rasterized layers at those same geometric samples;
  2. the actual original-source full and independent-part Blender Cycles 32spp V18h emission-only render outputs at those pixels.

At a *single complete untied* source ray sample, closest face is one of G or R. Its **joint** premultiplied RGB is the fixed emission color of the nearest face, alpha 1. The independently ray-rendered G and R take their respective nearest faces separately, alpha 1 each when present, and are blank otherwise. Accumulate all sixteen valid samples into a pixel **after** joint per-ray depth compositing. Also calculate two **postfiltered** G/R OVER orders (they share the same alpha `a_G+a_R-a_G a_R` and may fail because G/R geometric coverage is correlated). Neither rendering method gets to choose a group from the V18i image error mask.

For pixels with UNKNOWN ray ties, incomplete stacks or invalid camera, output abstention; do not equate no geometry in a 4×4 grid with guaranteed source alpha absence. Geometry-only two-color images represent opaque source geometry with deliberately overridden shader; they do NOT claim to reproduce original source paint or the exact Cycles random 32spp pattern.

## Frozen sources and candidate selection (before run)

Sources **CAT** and **TRICERATOPS**, original immutable V14 job run `42075029264-1`, original pose ordinal 2 (real authored pose frame), original views `0,2,4,6` at 192×192. Geometry: original V14 teacher posed world vertices, exact rest faces, frozen 8-view original camera bundle; original source exact SHA declarations checked. Frozen V18a `42097722762-1` thresholds `0.045,0.10,0.18`. Exactly the V18h original source `42120479356-1` emission-only 40-frame images and V18h independent court are parents (Cat `305a518f238de91903128fb13fcccac5f2179fa944e9281a6a3cbc0f7b756ee2`, Triceratops `6e493f531bc6e2a914b621e15bb8aceb8e3de62f75515a24261ad3a1656d73ff`). V18j exact court SHA for geometry lineage: Cat `81dc4101d585b35d054972af5e14ab1df5c3192436829f71fe758dca8609584f`, Triceratops `117dd9a3be10c3864bea21b3ccabdf2ec8ad042212c9b6e07d8f49b0971592e5`. The fixed V11 anchor target source-face ordinal zero is inherited from V18h manifest; no new target is cherry-picked.

For each original view, source-face centroid projection creates source-geometric pixel candidates. Per threshold and view, select at most 32 all-source candidate pixels, 16 group-source candidate pixels, 16 complement-source candidate pixels, and the four cardinal neighbors of the first two group and first two complement candidates. All selections are deterministic and **never condition on source RGBA, V18i error class or V18j selected failure coordinates**. Duplicates across group thresholds and views are cached by identical original view/pixel for the expensive ray queries. Camera views 0,2 are designated exploratory/reporting and views 4,6 are designated *locked evaluation cameras* (their pixels still selected by the same deterministic source-geometry policy); no proposal tuning or fitting may occur between them.

Fixed 4×4 offsets: `[-.375,-.125,.125,.375]` pixel in each axis (V17b). No fitted K. No post-hoc threshold selection. Entire K=1 group with empty complement is flagged VACUOUS regardless of reconstruction error.

## Independent tests and interpretation

For each selected original source camera/pixel/threshold, recompute:
- full joint subpixel geometry RGB/alpha vs both postfiltered candidate orders, in premultiplied linear RGBA;
- the same two posterior orders from actual V18h Blender source `FULL/ISOLATE/REMOVE` RGBA as a reference;
- **joint-sample** and **postfilter** errors relative to actual source V18h FULL linear premultiplied RGBA;
- order-invariant alpha miss and whether two different G/R nearest faces occur among subpixels;
- geometry rays with G→R→G or R→G→R depth alternation;
- qualifying/unknown ray count; `UNKNOWN` if any source ray tie/incomplete or non-finite original image;
- separate measurement for exploratory views 0/2 and frozen evaluation views 4/6.

This is a source-geometry **mechanism canary**, not guaranteed exact Cycles verification. A better match of sample-aware geometrical pixels to actual Cycles compared with postfiltered 2-layer OVER would favor sample-aware source rendering, but observed differences can include differing 4×4 vs Cycles random samples and color transform/antialias. It never proves artist part labels, global layer order, commercial rendering fidelity or generic unseen morphologies.

The independent court will re-open **actual original source geometry + original source renderer bytes** and recompute all rays and numeric evidence; not merely compare self-declared summary counts. No geometry first hit is converted to a learning input or a `SAME_DRAWABLE` / `DIFFERENT_DRAWABLE` edge.

All outputs `graphics_owner_authorized=false`, `iris_training_authorized=false`, `tessa_training_authorized=false`, `training_authorized=false`, `materialization_authorized=false`.

## Independent original-source outcome — completed, 3/3 PASS

Run `42128783773-1`, https://github.com/merynz/RealSaS-OPT/actions/runs/42128783773 : redteam 9/9 PASS, CAT original source PASS, TRICERATOPS original source PASS. Independently re-opened original source assets and reran all selected geometric 4×4 samples, SHA-checked readback to private Drive.

- Cat: 744 source threshold/view pixel rows, **3,968 unique geometric subpixel rays**; private independent court SHA `0019f17e7f1c08fcdad5c530957a4f7ae5f80bbbfe23ecca083be51ab5aa5535`.
- Triceratops: 751 source threshold/view pixel rows, **6,416 unique geometric subpixel rays**; private independent court SHA `48d9142ac948d4805ea183b84199e9a13a9a3e841b514acdea1cc17364d01545`.
- Every queried source ray was complete/untied in both subjects. Repeated threshold/view rows reuse immutable 4×4 original geometry rays; never treat them as independent rays or independent subjects.
- The original private source-pixel dataset was retrieved and aggregates independently cross-tabulated after the hosted court passed; no raw source mesh/source-face/private raster pixel data committed to Git.

### Critical holdout camera 4/6 result — Triceratops, frozen threshold 0.10

| Evidence over the 149 preselected original pixels | Count |
| --- | ---: |
| Actual V18h original Blender `FULL` versus Blender `G OVER R` / `R OVER G` both fail at tolerance 0.025 | **16** |
| Sampled original geometry shows opposing nearest G/R surface across subsamples | **14** |
| At 4×4 source-ray sampling, joint-then-filter differs from best filter-then-OVER by more than .025 | **11** |
| Geometric joint-then-filter is closer to **actual** V18h full Blender source than Blender part-OVER | **15**, all 15 were among the 16 failed Blender postfilter pixels |
| Failures rescued inside error tolerance .025 by geometric joint sampling | **3 / 16** |
| Pixels for which the 4×4 geometric joint-then-filter is worse than actual Blender postfilter | **131 / 149** |
| Pixels for which it is better | 15 / 149 |
| No numerical difference (up to 1e-10) | 3 / 149 |
| Actual Blender independent-layer postfilter errors above .025 | **16 / 149** |
| Geometric uniform-4×4 joint errors above .025 when compared to original V18h full render | **32 / 149** |
| Average pixel max-premultiplied-RGBA error, actual Blender layer OVER | **0.01850** |
| Average pixel max-premultiplied-RGBA error, geometric joint-then-filter | **0.02536** |
| Average pixel max-premultiplied-RGBA error, geometric filter-then-OVER | **0.03515** |

This is a **two-sided result**:
- The *same-geometry, same-samples* counterexample is real: joint 4×4 depth compositing and postfiltered 4×4 alpha OVER can differ. Local source nearest-surface order is not necessarily one uniform layer order. On the 16 real V18h layer-OVER-failing pixels, the geometry-first estimate is closer on 15; this gives genuine motivation to preserve K-hit/sample-aware source geometry in Graphics Evidence IR.
- The **unsolved** source appearance obligation dominates global quality: a 4×4 deterministic geometric sample image is **worse on 131 other (previously accurately composited) pixels**, and doubles the number of pixels above the declared 0.025 source error bound from 16 to 32. Geometric sample compositing is NOT an accurate substitution for 32spp Cycles. Do not present the 15 improved pixels as general success, or blame every residual on a specific stochastic filter until controlled repro qualifies it.

For further discrimination *within the 16 failing heldout pixels*, geometric joint-then-filter is closer than geometric filter-then-OVER on 11, which limits how much of the 15 apparent gains can be attributed to the *actual order-of-operations change* rather than different sample pattern or other approximations. Only 3 become source-tolerance PASS. One of the 16 gets worse.

### Exploratory camera 0/2 replication and controls

| Original source/threshold | Pixel queries | Real Blender independent-layer failures | 4×4 sample-aware improves Blender on prior failing pixels | 4×4 sample-aware all-pixel failures |
| --- | ---: | ---: | ---: | ---: |
| TRICERATOPS 0.045 exploratory | 118 | 4 | 3 | 17 |
| TRICERATOPS 0.10 exploratory | 150 | 9 | 8 | 23 |
| TRICERATOPS 0.18 exploratory | 108 | 0, **K1 vacuous** | 0 | 15 |
| TRICERATOPS 0.045 locked views | 118 | 4 | 3 | 18 |
| TRICERATOPS 0.10 locked views | 149 | 16 | 15 | 32 |
| TRICERATOPS 0.18 locked views | 108 | 0, **K1 vacuous** | 0 | 15 |

CAT is an **uninformative control for failure** under this original frozen V18h target group: zero actual Blender two-layer source failures for 124 original pixels per threshold across exploratory views and another 124 per threshold in locked views. Yet the 4×4 geometric approximation generates 15 errors above tolerance in the locked view group (each threshold). This directly warns against claiming that geometry-only 4×4 sampling preserves already-good 2D source appearance.

### Revised research decision (after FAIL-CLOSED quality evaluation)

1. **Do not promote V18k joint 4×4 render to a production renderer.** It is an explanatory geometry oracle and a plausible structural ingredient, but its source fidelity is insufficient in this controlled test.
2. For operational 2D presentation, retain original source 3D surface addressing, visibility/partial source-ray order, and the original appearance authority. Introduce source-matched *common sampling / filter policy* where full, isolate, remainder and sample-first renderer share the exact sampling domain. Either prove exact Cycles sample/coverage linkage or use a self-consistent independent raster reference followed by **separate verification against original artist presentation**; do not conflate these tests.
3. Reconstruct **same-source** 2D compositing on heldout motions/views with opacity, overlap, source texture, depth ties and masking; alpha and RGB source appearance must pass independently.
4. Only then audit the **editable layer operation and group boundaries** via original-source heldout edit consequences. No V18k sample ray becomes a must-link/cannot-link or IRIS student input.
5. Fail-closed status remains: original graphics-owner labels UNKNOWN; IRIS/TESSA/materialization/training authorizations FALSE. `main` untouched and PR #343 draft/open.

## Candidate downstream only

After source court results: design partial-order / coverage-aware Graphics compiler primitive and independently audit heldout edit consequences. Then test anonymous variable-cardinality grouping baseline (FH, signed Mutex Watershed / Multicut, or StdGEN/Bunraku-inspired modular surface proposals) with source-derived admissible constraints. Never reward K=1 as editability.
