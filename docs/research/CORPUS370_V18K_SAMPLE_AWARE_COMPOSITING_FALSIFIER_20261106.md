# Corpus370 V18k — geometry-precommitted sample-aware compositing canary

Date: 2026-11-06 (Türkiye local). **PREREGISTERED; THIS DOCUMENT PREDATES NEW ORIGINAL-SOURCE V18K MEASUREMENTS.** Draft PR #343, branch `corpus370-v18b-reactive-stain-source-edit-court-v1`. Do not equate any prospective PASS with drawable identity, original art alpha or model training authority.

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

## Candidate downstream only

After source court results: design partial-order / coverage-aware Graphics compiler primitive and independently audit heldout edit consequences. Then test anonymous variable-cardinality grouping baseline (FH, signed Mutex Watershed / Multicut, or StdGEN/Bunraku-inspired modular surface proposals) with source-derived admissible constraints. Never reward K=1 as editability.
