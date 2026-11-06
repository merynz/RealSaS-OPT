# Corpus370 V18n — source material/UV replay and depth-ordered editable presentation

Date: 2026-11-06, Istanbul. **ORIGINAL SOURCE MEASURED; INDEPENDENT SHA-BOUND RECOVERY COURT 3/3 PASS. No artist owner/model authority.** PR #343 research branch only; no main merge, no graphics owner or model teacher promotion.

## Fixed problem and lineage

V18m isolated the renderer's depth-order decision from the V18l FULL image target. Its independent **original source**, SHA-bound `42135271057-1` recovery courts passed 3/3 for CAT + TRICERATOPS: all 1,179,648 sample-pixel observations per subject qualified under eight identical Cycles source subset indices and four original cameras, with zero oracle-free original controlled **opaque two-color emission** reconstruction mismatches. TRICERATOPS had 18,342 original G/R simultaneously present rays; full-image erroneous pixels (two-layer postfiltered / fixed-order sample-first / independent depth source) `216 / 85 / 0`. This establishes a renderer mechanism only, not anonymous graphics ownership, source UV/alpha fidelity, artist editability, or IRIS prediction.

Frozen original parent geometry, authored pose ordinal 2, original 192² calibrated view indices `0/2/4/6` and frozen V18a `0.10` group (selected by original V11 face target 0) remain unchanged. Original source V14, V18a, V18f physically edited source and V18g rendering source courts are SHA-verified as in V18m. V18m source-depth independent recovery court SHA anchors:

- CAT `a39879e4558139ca2f418c1b4f33c581f0f6ced9c7fa4a89b5faed425266e889`
- TRICERATOPS `3981af638b632a512b3fde4ac6a75d5d0e2d8a831a81f5becb017af26233ac99`

V18l common Cycles sample source courts remain fixed: CAT `cce24e65850308ed0c27bcb63f33ec060ae7242fa57b27f4c82dc075e8be9a17`, TRICERATOPS `3245c11e49bc3e39b3209eda2ff771b39ec0269b7f66c45949476d8cb557333d`.

## One controlled independent new-source observation

Use **actual imported original source meshes**, plus the existing original posed 3D triangle-soup candidate with **original face-material slot IDs, source UV loops and sealed material-transform policy**. Do NOT replace the source material with constant red/blue emission. The source appearance policy is the **admitted SourceAppearancePolicyReplay**, not necessarily raw/untransformed authored artist shader; any source-original claim must name this distinction.

Frozen Cycles setup remains: samples 32 configured, seed 0, adaptive off, sample subset length 1, sample offsets `0,4,8,12,16,20,24,28`; pixel BOX filter width `.01`; 4 original cameras, no V18i/V18j error-selected pixels. For each source camera/sample render **four physical source cases** in the exact same scene:
1. `SOURCE_NATIVE`: original intact source meshes, soup hidden;
2. `SOUP_FULL`: original source triangle soup intact, native meshes hidden;
3. `SOUP_ISOLATE`: physically keep only frozen V18a group faces in soup;
4. `SOUP_REMOVE`: physically delete frozen group faces in soup, retain complement.

8 × 4 × 4 = **128 real original Cycles sample renders per subject**. The original SOURCE_NATIVE and SOUP_FULL are **heldout targets only**. Predicted compositing receives just the isolated G/R appearance source RGBA and original V18m isolated G/R geometry View Z depth bytes from sealed `42134142993-1`. The classifier never receives source FULL or artist layer identity.

### Two independently scored obligations

**A. Source appearance replay parity:** `SOURCE_NATIVE` vs `SOUP_FULL` in linear premultiplied RGBA per Cycles source sample, per pixel and all original four source cameras, including alpha. This tests whether source material/UV/normal/carrier is retained through source-triangle representation. Report every mismatch, even when depth-compositor succeeds at reproducing its own soup. A SOUP_FULL vs SOURCE_NATIVE mismatch must not be hidden by a lower downstream loss threshold.

**B. Two-source appearance compositing:** take `SOUP_ISOLATE` and `SOUP_REMOVE` actual source-material RGBA, choose G-first or R-first **solely using V18m source-geometry per-index View Z**, and apply exact premultiplied Porter-Duff OVER **before eight sample colors are averaged**. Compare against both independently rendered `SOUP_FULL` and `SOURCE_NATIVE`. Measure sample-level errors independently of eight-sample pixel mean; per-sample opposite errors may cancel in mean and must not be certified as correct. Also compare to the best fixed-order sample-first baseline and postfiltered layer OVER; count gains AND regressions.

Source material alpha may be nonbinary; geometric V18m alpha denotes occupancy, not actual paint opacity. A pixel containing material alpha zero *can* still have an original source 3D depth hit. Depth/appearance sample alignment only forbids **visible material where original source geometry missed**, not invisible material where geometry hit. Any inconsistent visibility, invalid depth, tie `|zG-zR|≤0.0002`, nonfinite RGBA or ambiguous camera -> FAIL/UNKNOWN, not forced same/different graphic component. A predicted G/R depth ordering is view/sample-local, not a global 2D layer order.

### Anticipated failure modes (do not conceal or call MODEL PASS)

- Native/soup original material parity fails: geometry/UV/tangent/shading/carrier source representation loss; must repair source first, even if SOUP_FULL is self-composable.
- Native/soup parity passes but G/R compositing fails: alpha blending/transparency/stochastic secondary rays, source material shader or depth policy invalidates simple two-plane Porter-Duff OVER; inspect without relaxing threshold.
- Both pass under opaque simple source materials: still not proof for arbitrary artist alpha, transparent/glass, variable K, broad morphology, heldout operational edits or unseen IRIS output.
- A source with zero material diversity/only binary alpha is a **control**, not a proof of alpha generality. Report source material opacity strata. Cat's tiny frozen V18a group is not a meaningful nontrivial drawable success.
- All results use original teacher source authority and source-private material/geometry IDs; they are never student model inputs. `SAME_DRAWABLE`/`DIFFERENT_DRAWABLE` remain UNKNOWN.

## Actual original source results — independent V18n court

**Immutable physical source render acquisition:** workflow `42137097177-1` (https://github.com/merynz/RealSaS-OPT/actions/runs/42137097177) acquired 128 real Cycles original native/full-soup/isolate/removal renders per subject. The first 1/3 source workflow completed contract tests, and **both original source jobs returned failure only at a copied V18l status literal** in the V18n output manifest; the original full source RGBA NPZ bytes and manifests were uploaded by the original trap and were never changed. The producer literal is now corrected in the same PR for future deliberate replays; expensive original source capture workflow was frozen to manual dispatch rather than automatically rerendering these SHA-sealed frames.

Exact original raw manifest SHA:
- CAT `79e3044bd030dea0788388573df7e75687a0454bd06bae7f61b4e43808c1eca6`.
- TRICERATOPS `7530def09abace3936ca9715460e78157254970850c733aa7e0f9dbd37ab2913`.

**Source recovery was not a rerender or changed scientific acceptance threshold.** `tools/corpus/recover_corpus370_v18n_original_material_status_receipt.py` verifies an exact source manifest SHA, exact original source RGBA NPZ SHA, camera set, material policy, group/arms and the negative authority flags, then changes precisely one manifest field (`status`) into the canonical V18n literal, writing both original/canonical SHA and a private transformation receipt. Every other manifest byte-value and the original 128-frame source pixel NPZ are immutable. New independent workflow `42138873648-1` (https://github.com/merynz/RealSaS-OPT/actions/runs/42138873648) **3/3 PASS**: redteam contracts, CAT source court, TRICERATOPS source court. The private reports were SHA-readback checked on Drive:
- CAT independent source court `99425ef370650913e5959d17ad6e02d59dc3c47ea2cb1e789f0c56af7bab319f`.
- TRICERATOPS independent source court `5c90e874a21bb549f0947c0745133f5d57d3afa11a4abcccd780cf59453b2f6b`.

Each original subject covers 4 cameras × 8 exact original Cycles subset offsets × 192² pixel instances = **1,179,648 sample-pixel comparisons**, 2,359,296 combined across two original subjects. Every sample was qualified with 0 depth ties, invalid depth or UNKNOWN output; no source material visibility without a geometric source hit.

### Original source appearance parity — SOURCE_NATIVE vs SOUP_FULL

Both subjects had **0 RGBA pixel-sample errors at the preregistered .025 threshold** across all four source cameras. Cat's raw source-vs-soup maximum absolute premultiplied RGBA difference was ~`5.4×10^-7` (far below tolerance); Triceratops matched to 0 in the independently scored source-domain output. Hence the **particular** original UV/material/pose triangle-soup representation does retain the admitted source appearance policy on these measured subjects and views.

This is NOT equivalence to all raw artist materials or original full 32spp AA film: the source policy's Principled shading may already be converted into source color-emission with retained alpha, and source sample subset + BOX width .01 differ from historical V14/V18h original film.

### Original source sampled material compositor

| Source subject / original camera | Original soup postfiltered G/R OVER failed 8-sample pixels | Fixed-order sample-first G/R OVER failures | Independent original-source depth-ordered sample-first failures | Original source G/R geometry both-hit sample instances |
| --- | ---: | ---: | ---: | ---: |
| CAT v0 | 0 | 0 | **0** | 5 |
| CAT v2 | 0 | 0 | **0** | 0 |
| CAT v4 (locked) | 0 | 0 | **0** | 5 |
| CAT v6 (locked) | 0 | 0 | **0** | 0 |
| TRICERATOPS v0 | 7 | 0 | **0** | 7,966 |
| TRICERATOPS v2 | 58 | 0 | **0** | 1,209 |
| TRICERATOPS v4 (locked) | 9 | 1 | **0** | 7,954 |
| TRICERATOPS v6 (locked) | 62 | 2 | **0** | 1,213 |
| **TRICERATOPS total** | **136** | **3** | **0** | **18,342** |

A predictor receiving only **source-isolated G/R material RGBA and exact V18m original source Cycles G/R geometry-depth images** correctly selected local source frontmost and applied premultiplied OVER **at each same Cycles sample before raster filtering**, without looking at source FULL or source-native colors for ordering. Across all original qualified samples, its source soup and source native **sample-level and 8-sample pixel-level errors at tolerance .025 were zero**. All 136 old postfilter source residual pixels and 3 fixed-order source residual pixels were recovered; **0 new previously passing pixel regressions** were counted.

Compared with V18m's fixed red/blue emission results (216 postfilter and 85 fixed sample-first failure pixels in Triceratops), these original source appearance replay results show fewer postfilter/fixed-order failures (136/3). **The counts must not be pooled as identical distributions**: V18m and V18n use different color/material policies and different actual source RGBA; only each source-conditioned experiment's baselines are directly comparable.

### Explicit limitations that matter for the next experiment

- Both measured subjects' `1spp` source-native alpha had **0 intermediate-alpha sample instances**; original material alpha was binary 0 or 1 in the tested cases. This does NOT validate semitransparency, alpha blending with continuous opacity, glass/transmission, additive effects or multiple transparent hits.
- Cat's fixed two-face G group has **no observable source paint consequence** under the tested edited source state, so is only a carrier/renderer control, not a proof of individually editable graphics.
- Triceratops's physically removed group has a clearly observable image consequence in the raw original Cycles material data (source-soup versus group-removed intervention; raw inspection counted 109,401 differing source sample-pixel instances across the four cameras). But the independent owner/editor court still does **not** prove a valid generic drawable component or an artist-intended part partition.
- Two source groups only, original teacher geometry/depth given, original material-policy replay not untransformed artist scene, one authored pose and four cameras. Neither arbitrary anonymous variable-K graphics, heldout editor edit independence, motion/world appearance or unseen IRIS predictions are yet qualified.
- No `SAME_DRAWABLE`, `DIFFERENT_DRAWABLE`, materialization authority or model training target was promoted.

**Next scientific gate:** a source cohort with demonstrably **nonbinary/material alpha** or true alpha layering, more than two anonymous groups under one fixed original source, original physical **editable heldout effects** (move/delete/reveal) and source appearance parity under the same original Cycles sample domain. Build a generic K-hit view-local compositor and camera/pose-stable group identity, rather than another hand-tuned per-subject segmentation threshold. Compare anonymous grouping algorithms only against these independently qualified operational edits.

### Deliverables and authority

Independent same-index source court with actual original Blender NPZ SHA + exact original V18m independent court SHA, separate original source/soup/compositor alpha/RGB metrics by reserved evaluation cameras 4/6 vs exploratory 0/2, **including all pixels not selected from prior errors**. No stage automation moves to IRIS/TESSA; no corpus subject readiness count changes; PR remains draft.

Code:
- `tools/truth_foundry/blender_corpus370_v18n_original_material_samples.py`
- `tools/corpus/court_corpus370_v18n_original_material_sample_replay.py`
- `tests/corpus/test_corpus370_v18n_original_material_sample_replay.py`
- `tools/corpus/hosted_corpus370_v18n_original_material_samples.sh`
- `.github/workflows/corpus370-v18n-original-material-depth-replay.yml`

**Conclusion:** Real-source physically edited UV/material-policy replay and sample-local original depth-order compositor passed independently on Cat/Triceratops. Artist drawable ownership, nonbinary alpha handling and unseen student inference remain UNKNOWN.
