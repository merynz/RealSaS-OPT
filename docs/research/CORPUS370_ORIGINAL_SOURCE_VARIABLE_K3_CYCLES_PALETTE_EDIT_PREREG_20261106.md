# Corpus370 V18 — Original-source three-palette same-Cycles-sample edit preregistration

2026-11-06. Draft PR #351. **NO GRAPHICS OWNER, IRIS/TESSA, MATERIALIZATION OR TRAINING AUTHORITY REGARDLESS OF OUTCOME.**

## Previous results

V18m/V18n source-private two-group source per-Cycles-sample compositor passed original Cat/Triceratops. New variable-K composer has independently passed original-source K2 regression with 1,179,648 Triceratops sample-pixels (all zero mismatch), but original source K3 is NOT yet tested. V1 physical palette test (V0/2/4/6, 16spp) FAILED one apparent collateral pixel. V2 (V1/3/5/7, 3x3 subpixel stencil) FAILED four. Both FAIL records remain unchanged. Analytic source triangles found a diagonal palette boundary crossing a supposedly "interior" V1 sample; exact Cycles sample coordinates were not recovered.

## Frozen V3 test

Use original SHA-locked Triceratops FBX ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a; original source rest world, camera bundle SHA e73917f650ee2dd925efe68a88e2fafdd5f99dbc5b7c410b7de0532ecf66bb0d and independent original material palette-facts SHA 8ff5534e4e8d7549cb9d419e4aea58511e0e042f9113e7d5807a8652a1b76edc. Exactly three source-qualified opaque constant Base Color groups across all 1334 original triangles. Same exact-color groups; no privileged bone/object labels, no per-subject threshold or artist drawable claims.

- Physically isolate each group by deleting other source triangles on separately cloned original mesh; no fake transparent BSDF deletion. Verify each original triangle appears in exactly one isolated clone.
- Original Blender **4.5.14 Cycles CPU**, 32 configured samples, seed 0, adaptive off, sample subset length 1, sample offsets (0,4,8,12,16,20,24,28), BOX width .01, 192x192 RGBA linear EXR.
- Original cameras V1,V3,V5,V7 (already inspected by V2 under a different sampler; not globally unseen views). Same source rest pose; do not call this heldout pose.
- Render each of **13 arms** on every identical camera/sample: original FULL (1), physically isolated group RGBA (3), source View Z depth of each isolated group (3), edited isolated group RGBA (3), and original edited FULL for each group (3). Total 4 cameras x 8 indices x 13 arms = **416 original Cycles sample renders**.
- Original imported opaque alpha/source Base Color-to-emission policy; restore original shader material state after every edit. Frozen channel recolor: component >=.5 becomes .15; otherwise .85, leave alpha unchanged. Every edited-full image is an independent evaluation target, not a predictor input.
- K=3 depth composer only receives same-source sample isolated RGBA + isolated original View Z depth, with edited isolate substituted for a chosen palette group. No FULL or edited FULL input to predict ordering. Camera-derived original world depth bounds are geometry-only.
- Independent court scores EVERY qualified original pixel sample, separately edit and baseline, and independently scores eight-sample reconstructed pixels. Premultiplied RGBA max-channel tolerance .025 (same V18m/V18n). Original material visible without matching geometry source hit FAIL. Depth ties <=.0002 and nonfinite depths UNKNOWN.
- Nonvacuity: >=100000 qualified sample-pixels, >=8 same-source pixels with physically present overlapping isolated groups, and each palette color edit must alter >=8 qualified source pixel samples. Each source camera and each edited-full consequence reported. No cherry-picking, no threshold retuning after viewing results.
- A real PASS is source-specific SAME-CYCLES-sample three-color edit and depth compositing proof. NOT generic segmentation, same_drawable/different_drawable truth, drawables count, source Cat texture editability, source alpha transparency, attachment, secondary motion, collision or multi-pose stable ownership.

All teacher-private original source materials/depth/geometry must remain in private Drive with immutable SHA lineage and independent court receipts. FAIL is preserved.
