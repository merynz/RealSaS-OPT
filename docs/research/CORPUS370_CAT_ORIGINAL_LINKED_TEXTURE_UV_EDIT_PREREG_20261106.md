# Corpus370 — Original Cat texture-UV edit operation preregistration

2026-11-06 · PR #351 source research only. No artist drawable truth, no model training authorization.

## Evidence and independently source-identified input

Original Kenney Cube Pets animal-cat.glb SHA 1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850, external Textures/colormap.png SHA 2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d; source rest camera bundle SHA 300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0 and source render policy SHA c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b. 684 original source triangles, 684 with linked Base Color texture and ZERO constant-only color-palette qualified faces. Source GLB has seven mesh primitives, one externally referenced RGBA texture 512x512, one source material. Its authored animations are removed ONLY in the separate original-source bind-rest derivative, never raw source.

## Preselected source UV operations, no rendered-image target leakage

Independently enumerate original GLB 2.0 triangle primitive POSITION/TEXCOORD_0/index tuples; for each source triangle compute normalized UV centroid. Bin into 8x8 original UV tiles, sort descending by triangle centroid counts then Y/X ascending, choose first three tiles containing at least 16 triangle centroids. Original source-only census gives (U6,V6) 292, (U6,V7) 246, (U7,V6) 63. These are anonymous UV editing ADDRESS REGIONS, NOT source art drawables, and may be spatially disconnected on the character.

Every tile is 64x64 exact source texture pixel addresses; v convention bottom-up, PNG pixel y top-down. Each tile is edited by inverting every RGB channel as uint8 (255-original_channel), alpha preserved; no pixels outside the exact selected tile may change in the decoded atlas. The image edit regions are distinct; image-space raster influence may overlap at texture mip/filter boundaries and is measured, not assumed zero. Source texture colors, GLB UV coordinates and source mesh data cannot be selected or modified using renderer success/failure pixels.

## Frozen physical source experiment

Original Cat bind REST, source material and texture linked, existing source unlit-alpha policy replay (not raw lit artist shading). Original camera views V1/V3/V5/V7, Blender 4.5.14 Cycles CPU, 32 configured samples, seed 0, adaptive off, one-sample subsets indices 0/4/8/12/16/20/24/28, BOX width 0.01 at 192x192 linear RGBA.

Render eight source full-mesh scene arms at every same camera and same sample: BASE, SINGLE_0, SINGLE_1, SINGLE_2, PAIR_01, PAIR_02, PAIR_12, JOINT_012. Total 4 camera views × 8 sample offsets × 8 arms = 256 original physically texture-edited source Cycles sample renders. No source geometry deleted and no material color replaced with a constant shader.

## Frozen independent court

Source GLB, image and source UV operation plan are SHA pinned. A separate original texture edit decoder must verify exact source-atlas RGB inversion and preservation of original alpha/unaltered texels. PAIR/JOINT source render targets must NEVER inform source tile selection.

At each same Cycles sample, compute predicted pair/joint source **premultiplied** RGBA using only original BASE and independently measured SINGLE_i source renders:

pred(S) = base + sum over i in S of (single_i - base)

This proves at most **source physical texture edit consequence superposition**, not predicting unseen edits without observed SINGLE results. Score each one-sample RGBA and eight-sample mean against physically rendered PAIR/JOINT full original-source targets at inherited premultiplied channel tolerance 0.025; require zero mismatches among ALL tested pixels (no masking conveniently near boundaries). Demand every SINGLE edit changes >=32 sampled source raster observations, and each original atlas source patch changes all 4096 tile texel addresses while preserving the rest and alpha. Report singleton image-change overlap, not assert visually disjoint drawables. Any unmatched source sample remains FAIL and private evidence preserved.

Training, materialization, owner ID, source pose2, transparent alpha, attachment, secondary motion, IRIS inference, unseen subject generalization all NOT QUALIFIED, regardless of outcome. All owner/training flags remain FALSE.
