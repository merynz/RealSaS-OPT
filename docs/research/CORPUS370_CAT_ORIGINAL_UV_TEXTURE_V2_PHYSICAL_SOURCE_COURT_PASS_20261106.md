# Corpus370: Original Cat linked-texture UV edit — V2 physical source PASS (2026-11-06)

Draft research PR #351. **SOURCE GRAPHICS EDIT CONSEQUENCE PASS, NOT ARTIST DRAWABLE OWNERSHIP OR MODEL TRAINING PASS.**

## Source and experiment

Cat (Kenney Cube Pets) is a deliberately different source appearance class from the Triceratops constant shader palette: all 684 source triangles use texture-linked Base Color, and zero are admissible to the original constant shader palette. Original GLB SHA 1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850; external source 512x512 RGBA atlas SHA 2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d; source camera SHA 300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0; source material unlit-alpha replay policy SHA c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b. This source replay is not arbitrary raw artist-lit shading.

The source UV-only edit plan V2 (SHA 8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b) selected three independently addressable 64x64 PNG regions from original GLB TEXCOORD_0 centroid counts only: (U6,V6) 292 source face centroids, (U6,V7) 246, (U7,V6) 63. Raw glTF has image TOP-LEFT UV origin. Edit RGB: 255-original uint8 per channel; original alpha and all other source texels preserved. No artist object names or source image render target pixels used to choose regions. Source UV regions are not artist drawables.

Four original cameras V1/V3/V5/V7 x 8 same Cycles offsets 0/4/8/12/16/20/24/28 x 8 source arms (BASE, 3 SINGLE, 3 PAIR, 1 JOINT) => **256 physically original-source full-scene Cycles renders**; 192x192 resolution, seed0, adaptive off, BOX .01, original rest pose and linked material.

## Immutable negative lineage

V1 original source physics [42259366611](https://github.com/merynz/RealSaS-OPT/actions/runs/42259366611) scientifically FAILED: all 3 source edits had zero visible consequence. Original raw glTF UV V was incorrectly inverted when mapping source UV tile to PNG atlas pixel Y, so original black unused pixels were edited. No additive-composition success was claimed. V1 court SHA 6cdd432ce3a3ceb64d9c88117847f99336d33d9891b483a95c8bbb2d0fa8b186. The [V2 preregistration amendment](CORPUS370_CAT_GLTFTOPLEFT_UV_ATLAS_V1_FAIL_V2_PRECOMMIT_20261106.md) fixed only source UV-to-PNG Y address, preserving original selected tiles, RGB operation, sample domains, heldout equation, tolerance and nonvacuity.

V2 source-plan SHA cohort [42260446130](https://github.com/merynz/RealSaS-OPT/actions/runs/42260446130) PASS. V2 first physical attempts 42262350923 and 42262890749 stopped BEFORE rendering due to the workflow mistakenly invoking the obsolete V1 PNG variant producer against the new V2 plan. No scientific result in these attempts. Correct source producer, physical renderer and independent court were run in [42263195370](https://github.com/merynz/RealSaS-OPT/actions/runs/42263195370).

## Independent actual source results

[Original independent Cat UV V2 source court](https://drive.google.com/file/d/1OcivCdBEmTK5jp02GGY0oLV7v-WPxOzk/view) SHA **90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08** reports **PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN**, failed obligations empty, original source alpha drift exactly 0. Original edit atlas manifest SHA 0e9a2fcf6a2c048eb6e99048793290b8865d802689f59b7c8c20bc6390cf479a, original render manifest SHA 8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1, and original source samples NPZ SHA d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114.

| Original source view | SINGLE0 changed sample-pixels | SINGLE1 | SINGLE2 | SINGLE0+1 shared visible-support samples |
| --- | ---: | ---: | ---: | ---: |
| V1 | 70,932 | 61,420 | 3,022 | 629 |
| V3 | 74,528 | 70,828 | 0 | 779 |
| V5 | 74,556 | 70,863 | 0 | 776 |
| V7 | 70,963 | 61,443 | 3,026 | 615 |
| ALL | **290,979** | **264,554** | **6,048** | **2,799** |

Sample-pixels are per-source-physical-edit sample incidences, not unique image pixels or source body parts. SINGLE2 is hidden/nonobservable in V3/V5. SINGLE0 and SINGLE1 do have 2,799 overlapping image-space influence samples although their atlas texel addresses are disjoint; do NOT claim each operation has disjoint projected coverage or is one true drawable.

Independent source court predicts PAIR/JOINT physical source FULL images by baseline plus SUM of independently physically rendered SINGLE minus baseline **in linear premultiplied RGBA**, using matching Cycles sample indices. Physical PAIR/JOINT FULL arms are heldout; they are never used to select atlas regions or calibrate predictions. All original pair/joint sample-level and 8-sample-mean mismatch counts at the inherited 0.025 max-channel RGBA tolerance are **ZERO** across 4 cameras x 4 heldout combinations. Maximum absolute sample difference is ~0.005161 (not bitwise exact). Each source atlas 64x64 operation mutates 4096 original texels, original texture alpha and all out-of-tile texels preserved. All globally preregistered nonvacuity edit gates >=32 pass.

## Typed implications for final 370 truth-family matrix (SOURCE candidates, NOT authority)

- APPEARANCE_CAA: 684 texture-linked source triangles, SHA bound to original source atlas / image shader replay. Only controlled source appearance replay, NOT general CAA or original lit artwork.
- CAUSAL_RESPONSE: 3 independent texture RGB edits, **561,581** summed edit/sample-pixel incidences across all 4 source cameras, and verified joint-source consequences. Color intervention, NOT motion/mechanics response.
- VISIBILITY_OCCLUSION: this test does NOT independently qualify layer ordering. Existing Cat V18n source depth evidence remains a different sample/pose policy, cannot be silently joined.
- RAY_COMPOSITING: no isolated original K-layer compositing claim from V2; this is an independent additive texture edit consequence, not geometry depth sorting.
- Source drawable partition, artist part count, SAME_DRAWABLE/DIFFERENT_DRAWABLE edges, pose-persistent attachments, secondary motion, semitransparent alpha, single-view IRIS inferability: UNKNOWN, 0 training labels.

All source family final authority is PENDING. IRIS/TESSA/materialization/training_authorized FALSE. No main merge. Next bounded target: preserve source-provenance and per-view effect overlap in a typed Cat Graphics Evidence IR / independent source court for the existing final truth-family admission bridge, and investigate real pose/attachment boundaries separately rather than treating UV tiles as part IDs.
