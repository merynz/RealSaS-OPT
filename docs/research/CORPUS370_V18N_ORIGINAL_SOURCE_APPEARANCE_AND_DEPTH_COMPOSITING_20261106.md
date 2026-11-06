# Corpus370 V18n — source material/UV replay and depth-ordered editable presentation

Date: 2026-11-06, Istanbul. **PREREGISTRATION — BEFORE ANY V18N ORIGINAL SOURCE MEASUREMENT.** PR #343 research branch only; no main merge, no graphics owner or model teacher promotion.

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

### Deliverables and authority

Independent same-index source court with actual original Blender NPZ SHA + exact original V18m independent court SHA, separate original source/soup/compositor alpha/RGB metrics by reserved evaluation cameras 4/6 vs exploratory 0/2, **including all pixels not selected from prior errors**. No stage automation moves to IRIS/TESSA; no corpus subject readiness count changes; PR remains draft.

Code:
- `tools/truth_foundry/blender_corpus370_v18n_original_material_samples.py`
- `tools/corpus/court_corpus370_v18n_original_material_sample_replay.py`
- `tests/corpus/test_corpus370_v18n_original_material_sample_replay.py`
- `tools/corpus/hosted_corpus370_v18n_original_material_samples.sh`
- `.github/workflows/corpus370-v18n-original-material-depth-replay.yml`

**Success is falsifiable**: no promotional claim until actual original-source physical edit replays and independent court pass; even then author drawable and original raw shader fidelity remain UNKNOWN.
