# Relations × Graphics — original Cat source-face ↔ pixel ↔ UV-intervention truth V1

**2026-11-07 · Source-only research stacked on PR #362 · original Cat original-source private bytes · no product/training authority.**

## Question

The independently courted original Cat V18s geometry/UV/mechanics relation evidence and PR #351 original Cat physical UV edit source samples share one **original Cat source SHA**, but the previously published physical edit ledger preserved effects at **image sample-pixel aggregate addresses**, not at original per-face source addresses. This was a real epistemic blocker.

We do **not** solve it by a learned pixel segmentation, source image thresholding, a convenient drawable grouping, or by pretending Cycles random subpixel offsets equal the older deterministic 4×4 geometry rays.

Instead, this V1 experiment asks for a **sound, deliberately incomplete subset** of original face↔sample-pixel edit witnesses that can be certified *without knowing exact source Cycles subpixel coordinates*.

## The geometric entire-pixel witness theorem

Let \(Q_{xy}\) denote a pixel's normalized orthographic camera square, enlarged uniformly by a fixed guard \(g=0.05\) pixels. Let \(\Pi(f)\) be the original source triangle \(f\) projected by the **exact same immutable original 8-camera bundle** used for physical source Cycles V2. Let \(z_f(q)\) be the affine source ray depth of triangle \(f\) at a screen point \(q\) inside its projection.

A source face \(f\) has a conservative **all-subpixel first-hit certificate** for \(Q\) if:

1. **Complete coverage:** all four corners of enlarged \(Q\) have strictly positive barycentric coordinates inside \(\Pi(f)\), with margin \(10^{-7}\). Since both \(\Pi(f)\) and \(Q\) are convex, \(\boxed{Q\subset\Pi(f)}\).
2. **Source camera clipping:** depths at all four corners lie strictly within the original source camera near/far clip, with \(10^{-5}\) depth-unit margin.
3. **Conservative exclusion of every other original source triangle:** for every \(g\ne f\) whose *projected triangle AABB* intersects \(Q\),
   \[
   \boxed{\max_{q\in Q} z_f(q) + 10^{-5}
     < \min_{v\in\operatorname{vertices}(g)} z_g(v).}
   \]
   This deliberately uses the **minimum depth of the WHOLE other triangle**, not a target-peeking pixel-depth oracle; an AABB false positive can only **remove** a certificate.

Then **for every orthographic camera ray through every \(q\in Q\)**, the nearest source triangle is \(f\), provided the archived original source triangulation includes all actually rendered surfaces and same source camera geometry is used.

**Proof:** Coverage is sufficient by convexity. Depth is affine on each projected triangle and maximized over \(Q\) at a corner. For any other source triangle \(g\) that could intersect a ray through \(Q\), its projected AABB necessarily intersects \(Q\), and any point on \(g\) has depth at least its vertex minimum by affine interpolation. By (3), the candidate \(f\) has strictly smaller depth everywhere. The original camera clip condition holds by (2). Thus \(f\) is the unique first intersection on every ray through \(Q\). QED.

**Boundary:** This proves the geometric face for ANY ray whose sample lies in the guarded pixel Q. We still condition its application to source Cycles on camera sample rays falling within that footprint; we are not pretending to independently know the random sample positions. The physical source court separately verifies original alpha and intervention outcomes for all eight stored source sample offsets, and fails closed for contradictions.

## The glTF UV-interior theorem

The original Cat source material has an independently SHA-qualified source glTF **TOP-FIRST** face-corner UV address map, the original scene is orthographic and the shader uses that imported atlas. Let \(u_f(q)\) denote face-corner UV linearly barycentrically interpolated at a screen point \(q\).

If all FOUR guarded pixel corners satisfy

\[
u_f(q)\in \operatorname{int}_{1\text{texel}} R_k
\]

for the SAME precommitted original 64×64 atlas RGB-invert operation \(k\), then the entire \(Q\) maps to that source tile because barycentric interpolation is affine and the tile interior is convex.

Combined with the first-hit theorem and originally SHA-verified source shader/alpha policy, the face/UV/edit address is a **sufficient locus candidate**. It does NOT assert that the edit must have a nonzero image effect: a grayscale midpoint texel, sampling/filtering, or invalid source renderer could make the edit unobservable. Observability must be checked from the ORIGINAL physical source edit samples independently. Likewise, a UV atlas tile is NOT an authored drawable/part.

## Physical empirical falsification obligations

For every emitted conservative witness with exact source face \(f\), original view \(v\), top-first pixel \((x,y)\) and edited source tile \(k\):

1. Reopen SAME original 256 real rendered Cycles source frames from PR #351 V2, with exact source GLB/teacher face order/UV map/camera/material policy and original independent court SHA pins. No new render, no adaptation of camera/atlas/threshold.
2. Convert only the original Blender EXR buffer row index \(y_{\mathrm{raw}}=191-y_{\mathrm{top}}\); no arbitrary warp, best-of-view or threshold tuning.
3. For each of the eight original same-index Cycles samples, compare **linear premultiplied RGBA** BASE against SINGLE_k at the original source tolerance \(\|\Delta\|_\infty>0.025\); require physically observable intended edit, no other SINGLE_i for \(i\ne k\), original sample alpha 1 within tolerance. This is **falsifiable**.
4. Independently remeasure each emitted witness with a **different direct all-original-face geometric algorithm**, not the producer's vectorized depth-envelope implementation. Compare actual source physical edit effects and counts; preserve SHA-locked report and independently courted private receipts.
5. All unwitnessed original faces/tiles/pixels are **UNKNOWN**, not negative edit/source identity labels.

The original physical source has *three* edited atlas tiles, but the geometric first-hit/UV-interior certificate may select only a subset. It does NOT need to fabricate any source occurrence for the missing tiles. The design measures source geometry/appearance causality, not 42 layers or potential puppet packet count.

## Source provenance

| Original source parent | Exact SHA-256 |
|---|---|
| Original Cat GLB | 1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850 |
| Native original source teacher with exact 684 face order and rest XYZ | c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603 |
| Independent original rest source mesh | 748609aca5ba7c509aa390ed65baeca7612b8879aa00409735da3a96758379cf |
| Native source-face corner UV witness NPZ | 7f3afdc4f0f848de3aa06bda2220151427d352368a0f235ca09d5d96f7834787 |
| Original source UV↔teacher GLB court | b966a3793c682cb06ec70341e659aabbded9343ace3d645c08643de40f2799d7 |
| Original 8-camera bundle | 300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0 |
| Original source material policy | c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b |
| Physical atlas RGB edit operation preplan | 8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b |
| Original eight-arm 256 source Cycles frames | d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114 |
| Independently original source physical edit court | 90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08 |

The original source rest array is **elementwise equal** to the native teacher source REST face index and coordinate arrays. It is NOT silently interchanged with the V13b authored pose0 baseline or the V3 pose14 example.

## Empirical pilot and acceptance gates

A local pre-registered *algorithmic pilot* (NOT yet a GitHub independent physical court claim) yielded the following conservative original source first-hit/UV tile candidates:

| View | Conservative source face↔UV edited pixels |
|---|---:|
| Source V1 | 109 |
| Source V3 | 193 |
| Source V5 | 175 |
| Source V7 | 131 |
| **Total** | **608** |

Preliminary source face identities cover **24 original faces**, including 555 pixels from atlas edit 0 and 53 from edit 1. **0** candidate pixels from original atlas edit 2 pass this deliberately strong full-pixel certificate. At the source 8 physical sample indices these 608 addresses carry 4,864 observed intended sample-edit effects and 9,728 complementary other-edit absence obligations. THIS IS NOT evidence edit 2 is absent in real artwork: earlier original physical source court independently measured 6,048 edit-2 effect samples in less strictly addressable pixels. The correct interpretation is **V1 sufficient certificate cannot cover edit 2**, thus its face-local status remains UNKNOWN.

Do not promote the pilot figures to independently court-admitted FACTS until the actual GitHub original private source court replays and SHA-readback passes.

## Scientific limitation and next gate

This is a **sound but incomplete source-fact localization**, not a complete whole-screen source face-to-sample ray truth. Whole-source pixel coverage, source alpha at non-qualifying rays, geometry behind originally opaque surfaces, general transparent original materials, all 684 source face effects, pose-dependent UV transport and unseen morphological genericity are OPEN.

It demonstrates a critical possibility: **some original per-face Graphics causal relations can be given proof-carrying source labels without knowing artist drawables, arbitrary pixel segmentation, inferred stochastic ray jitter or a trained network**.

Next expansion after independently qualified court: investigate original source edit-2 affected pixels with a smaller deterministic source-screen ray test or additional original source camera evidence; design a separately pre-registered physical ray-face AOV witness if the proof becomes insufficient, rather than weakening the current criterion to inflate counts.

## Code/CI

- Producer: tools/corpus/measure_original_source_face_pixel_uv_causality_v1.py
- Independent court: tools/corpus/court_original_source_face_pixel_uv_causality_v1.py
- Red-team tests: tests/corpus/test_original_source_face_pixel_uv_causality_v1.py
- Source workflow: .github/workflows/original-cat-source-face-pixel-uv-causality-v1.yml

No source-private face-index-pixel correspondence, original pixel arrays, source texture, model target or training authority is placed in the public repo.
