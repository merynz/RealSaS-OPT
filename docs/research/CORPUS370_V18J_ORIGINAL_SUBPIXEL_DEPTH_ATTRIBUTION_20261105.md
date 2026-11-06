# Corpus370 V18j — Original source 4×4 subpixel and depth attribution

Date: 2026-11-05. Status: **REAL ORIGINAL-SOURCE INDEPENDENT COURTS 3/3 PASS; NO GRAPHICS TRUTH OR MODEL ADMISSION**.  
PR #343 `corpus370-v18b-reactive-stain-source-edit-court-v1`; original run `42125868885` at https://github.com/merynz/RealSaS-OPT/actions/runs/42125868885 .

## North star

RealSaS compiles generic, persistent, source-authoritative *editable puppets*: 3D geometry and mechanics, and **2D presentation graphics** with transparency, independent editing, visibility, partial draw order, occlusion, and source appearance. The teacher need not recover artist's literal original layers, but it must support operationally correct edits. A V18a mechanical-response grouping is a proposal, not a drawable. Anatomy labels are excluded; StdGEN/Bunraku offer representational references, not truth.

## Frozen parent empirical proof

- V17b provides **4×4 fixed fractional orthographic source ray offsets** `[-.375,-.125,.125,.375]`, independently source-courted as a geometric coverage probe; source face hits are not exact Cycles alpha or artist ownership.
- V18e–f replaced false deletion-by-Transparent-BSDF with actual Blender face removal; original source/UV/material/no-op/empty/pose2 tests qualified. V18f 416 physical render frames; source Cat/Triceratops both independently PASS.
- V18g (original lit) proved group-and-remainder images do not always recompose via either fixed or pixelwise per-pixel Porter-Duff OVER order; K1 whole-character decomposition can give vacuous zero error.
- V18h original source two-color emission-only (without original artist shading), 40 actual renders/source, original pose2 / four views; independent source court 3/3 PASS. Frozen V11 anchor target 0 only; thresholds `.045,.10,.18`; original-source pose, camera, faces and V18a group labels are unchanged.
- V18i original source retroactive alpha-vs-RGB diagnosis, run `42123720590-1`, 3/3 PASS, exact private SHA readbacks: CAT `763c4bfebe78cc04dc37cd48aa1258b2b2ddffd0b222a158fc8fedf7d0d575d0`; TRICERATOPS `36b1bb856c24fa204dbb1cd3cef0a0cbbdfb2cc7fd5b8b31f59cd22286883bd2`. For Triceratops frozen `0.10` group, 457 render pixels defy both independent-render OVER orders: 172 violate **order-invariant alpha inclusion-exclusion** `aF=aG+aR-aG aR`; 285 keep alpha agreement but fail premultiplied RGB. Cat target 0 fails zero pixels under emission. K1 Triceratops .18 is not independently editable.

## V18j precise falsifiable question

Do original source-space, geometrically evaluated 4×4 subpixel rays display **intra-pixel mixed visibility**, opposing nearest G/R hit directions, or depth interleaving at pixels originally classified by V18i as:
- `ALPHA_FAIL`: original full alpha differs from postfiltered G OVER R predicted alpha, and both-order RGBA oracle fails;
- `RGB_ONLY_FAIL`: per-pixel two-order oracle fails without alpha identity violation;
- `VISIBLE_PASS`: full-source visible alpha, per-pixel two-order reconstruction succeeds.

**This is retrospective outcome-selected *explanatory* sampling.** It may diagnose mechanisms but is NOT a blind heldout test and cannot mint independent edit truth. For every original camera and frozen threshold, choose up to **five fixed raster-order evenly-spaced pixels per stratum**, with empty strata explicitly retaining zero count. Original framebuffers loaded from bpy EXR store **bottom-up** pixels; geometric source rays require the explicit **`top_down_y=191-framebuffer_y`** conversion. Without this, source-ray intersection agreement would be fabricated by mismatched coordinates.

At each fixed pixel, 16 source-posed ray-stack queries run through the existing `models/iris/truth/graphics_oracle_v1.py` geometric oracle, with camera offset from V17b's `offset_camera`. Every complete/non-tied ray's original source face indices and depth-sorted hits are recorded privately; relation to V18a anchor group is **anonymous G / complement R**, not author layer. A depth tie, incomplete camera stack, or no valid ray is an explicit UNKNOWN; a properly complete and empty ray is geometric no-hit, **not** absence of painted alpha.

If all 16 samples at a pixel are complete and untied, measure:
- geometric joint subpixel coverage fraction (any G or R);
- fractions of group and complement source coverage individually;
- hypothetical **postfiltered** alpha `pG+pR-pG pR`;
- difference from joint geometrical union fraction;
- opposing nearest G/R across subpixel rays;
- within-ray original mesh depth alternation `G→R→G` or `R→G→R`.

Crucial limitation: 4×4 uniform deterministic rays are **NOT Cycles's actual randomized 32-sample raster pattern**, and geometry hit does not imply opaque source material, texture alpha, physical source ownership, or correct artist draw order. Even if the geometric postfilter gap correlates with V18i's alpha error, this only suggests a mechanism; independent source raster replay is required for causal proof. No cluster SAME/DIF, signed ATTRACT/REPEL, or exact 2D alpha truth is minted from this diagnostic.

## Sealed original sources and reproducibility

Source Cat and Triceratops only. Frozen V14 job original `42075029264-1` with SHA-declared original native teacher posed world vertices, exact 8-view camera bundle and rest source faces. Frozen V18a original source labels run `42097722762-1`; V18h original source RGB(A) `42120479356-1`; V18i exact court `42123720590-1`, with the above expected exact SHA. Original author motion labels and materials are never student inputs.

The producer independently reopens original source geometry and actual raw RGBA, verifies source SHA + face/pose/camera correspondence and re-derives the **entire 12 case pixel-stratum census** before querying V17b's sample grid. The separate court reopens all these exact raw original assets and remeasures **every subpixel ray** and pixel stratum, rejecting altered source groups, camera basis, depth ties silently recast as negative data, unknown-to-negative coercion, duplicate sampled pixels, altered tolerance, outcome class tampering, missing rays or any training/graphics authority flags.

The original source 4×4 samples and source-faces remain private on Drive with SHA-readback of independent court; public Github only hosts source-agnostic test code and scientific results/aggregates. No Blender rerender is required. The original source host uses SHA-locked official rclone v1.75.2 transport (not Ubuntu apt), and unchanged V5 private source hydration to restore original teacher and camera bytes.

- Producer: `tools/corpus/measure_corpus370_v18j_source_subpixel_order_attribution.py`
- Court: `tools/corpus/court_corpus370_v18j_source_subpixel_order_attribution.py`
- Redteam: `tests/corpus/test_corpus370_v18j_source_subpixel_order_attribution.py`
- Source host: `tools/corpus/hosted_corpus370_v18j_source_subpixel_order_attribution.sh`
- Workflow: `.github/workflows/corpus370-v18j-original-source-subpixel-ray-attribution.yml`

## Confirmed original-source measurements — 3/3 PASS

Workflow `42125868885-1` was **3/3 SUCCESS**: unit/adversarial contract, original CAT, and original TRICERATOPS. Both original-source workers SHA re-opened exact original V14 posed teacher/camera/rest, V18a face grouping, V18h emissive 40-frame RGBA and exact V18i independent original-source court, then measured original 4×4 rays and separately remeasured all 16-ray stacks for every stratified pixel.

Independent remote court SHA receipts, checked by readback:
- CAT: `81dc4101d585b35d054972af5e14ab1df5c3192436829f71fe758dca8609584f`.
- TRICERATOPS: `117dd9a3be10c3864bea21b3ccabdf2ec8ad042212c9b6e07d8f49b0971592e5`.

**TRICERATOPS** — failure-selected explanatory pixels across all four source views and three frozen thresholds, with a cap of five per nonempty stratum/view:

| Original emissive V18i class | Pixels sampled | Re-courted valid geometric subpixel rays | Source pixels where 4×4 nearest G/R side flips | Source pixels with some G–R–G or R–G–R stack |
| --- | ---: | ---: | ---: | ---: |
| ALPHA_FAIL | 35 | 560 / 560 | **13 / 35** | **5 / 35** |
| RGB_ONLY_FAIL | 25 | 400 / 400 | **21 / 25** | **5 / 25** |
| VISIBLE_PASS | 60 | 960 / 960 | **0 / 60** | **1 / 60** |
| **TOTAL** | **120** | **1,920 / 1,920** | **34 / 120** | **11 / 120** |

All 1,920 Triceratops subpixel ray stacks were complete and untied. `RGB_ONLY_FAIL` shows **opposing nearest G/R source geometry across subpixels in 21/25** retrospectively selected pixels, versus 0/60 passing controls. This is a strong *diagnostic association* with mixed local visibility, not yet causal proof that mixed order explains full Cycles source RGB discrepancy.

**CAT** — V18h's single fixed group had no failing pixels, so only 60 `VISIBLE_PASS` source pixels could be selected across 4 cameras × 3 frozen thresholds. All **960 / 960** 4×4 geometric rays were complete/untied; **0 / 60** had opposing nearest G/R subsample sides and **0 / 60** contained 3+ alternating G/R depth runs. This is a healthy control, **not** validation of unprobed Cat groups or unseen morphologies.

### What this result justifies and what it does NOT

**Justified:** The minimal `triangle → group → independently rasterized PNG → OVER` representation is not sufficient for all V18h source pixels. Source geometry itself exhibits local subpixel-dependent nearest-layer changes strongly associated with V18i RGB failures. Evidence IR should preserve source-locus face and barycentric coordinates, per-sample geometric coverage, depth stacks, partial/view-local ordering, and validity/UNKNOWN masks, rather than inventing a global layer order or a semantic class.

**Not justified:** Exact 4×4 rays are not the 32 Cycles stochastic samples. A source geometric hit is not necessarily opaque art alpha. The selected pixels are from the *same original V18i outcome map* (explicit diagnostic selection), so 21/25 vs 0/60 is an **in-sample association**, not a blind heldout success. No graphics-owner relation, SAME/DIF, artist part K, attractiveness/repulsion edge, training admission or materialization permission follows. A mixed-nearest source ray can be a warning about the proposed **2-layer rasterization representation** without proving the group partition is artist-incorrect.

**Next:** Define source-private, typed **Graphics Evidence IR** with distinct fields for `SOURCE_RAY_GEOMETRIC_ORDER`, `GEOMETRIC_SUBPIXEL_COVERAGE`, `RG_BA_COMPATIBILITY_OBSERVATION`, `OBSERVABILITY_UNKNOWN` and `OPERATIONAL_EDIT_EFFECT`. Expose **no artificial signed edge** from this experiment. Then independently test a joint-subpixel compositor against source-controlled raster on a separately sealed input before promoting any graphics target.

## Interpretation gate after independent source courts

If alpha-stratum source rays have mixed G/R coverage, that motivates *sample-aware compositing* experiments, not an automatic cut to graph segmentation. If RGB-only failures show conflicting nearest order or multi-hit depth alternation, that motivates **view-local partial order and K-hit visibility**, not one total draw order. A single hit or null geometry result without exact alpha cannot license a real layer edit. A zero observable alpha/rgb-error stratum (as in Cat target0) is not generalization proof.

Only **after** source evidence is reconciled with actual source appearance/visibility may Graphics Evidence IR declare typed `GEOMETRIC_COVERAGE`, `SOURCE_RAY_HIT_ORDER`, `RGBA_EFFECT`, `UNKNOWN` and proposed operational constraints. No model/IRIS/TESSA head, training, materialization or Main branch authority is altered in this iteration.

This file starts as a **preregistered research record**. It must be amended with exact actual source court statuses and receipts, rather than presuming PASS while workers run.
