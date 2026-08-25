# P-V5 R256 — Hard Asset 36fb × 2 Styles Sufficiency V1 — Preregistration
Date: 2026-08-25
Status: FROZEN BEFORE OPTIMIZER STEP 1

## Parent evidence
The original 8 assets × 2 styles V1 gate is immutable FAIL. A preregistered +2048-step low-LR continuation is also immutable FAIL, but all 16/16 cells improved simultaneously and the worst-cell curve was still descending at the final authority checkpoint. Final continuation: 13/16 cells PASS, worst P95 `0.005740759451873588`. The dominant remaining blocker is `asset_36fb02305846592b1ecdf3d4` in both styles; `asset_76313.../ink_cel` misses by only `4.27e-5`.

Prior optimizer-zero residual microscopy localized the broad error to learned camera-forward depth rather than analytic screen-plane P, one broken view, or silhouette-boundary concentration. PatchMatch is therefore not admitted in this gate.

## Scientific question
With the certified R256 P representation and a fresh shared two-style learner, is the dominant hard asset itself learnable to `P_p95 <= 0.005` in both styles when given an evidence-backed adequate optimization budget?

This is a localization/sufficiency gate. It does not test family generalization and does not rewrite either 8×2 FAIL.

## Frozen membership
Asset: `asset_36fb02305846592b1ecdf3d4` (`FIT`). Styles: `cel_clean`, `ink_cel`. Exactly two asset-style cells. Both styles are present in every optimizer step. No style-ID conditioning or style-specific weights.

The hard asset is intentionally selected post-result because this is the preregistered localization branch after the continuation FAIL; it is not used as an unbiased product/generalization estimate.

## Representation / data
Unchanged true full-R P path. The learner predicts camera-forward scalar depth only; screen-plane coordinates, canonical yaw and native-image-derived `h_native` remain deterministic. N/U/Z frozen. Input derivative 512 RGBA -> one PIL bilinear resize to 256. Native 1024 only estimates `h_native`. 4096 deterministic visible raster-authority samples/view. Exact truth loci shared across styles. No augmentation. No `camera.json`. No TUNE/CAL/DEV/EXTERNAL/sealed consumption.

## Fresh initialization
Seed `20260825`. Fresh model; no checkpoint reuse from 8×2 or the old 76313 two-style PASS.

## Frozen adequate-budget schedule
MAIN: fresh AdamW, lr `3e-4`, betas `(0.9,0.95)`, wd 0, **2048 optimizer steps**.
TAIL: fresh AdamW moments on MAIN weights, lr `3e-5`, same betas/wd, **2048 optimizer steps**.

The longer tail is preregistered because the 8×2 continuation showed continued improvement through +2048 low-LR steps without a capacity plateau. It is not result-driven extension inside this gate.

Every optimizer step contains both style cells (`B=2`) and all 8 views. FP16 AMP forward; FP32 camera-forward SmoothL1 beta `0.01`.

## Evaluation schedule
INIT; MAIN `512,1024,2048`; TAIL `64,128,256,512,1024,1536,2048`. Selection minimizes worst-cell P95, then aggregate P95, then total steps.

## PASS / FAIL
PASS iff one preregistered candidate has **both** style-cell P95 values `<= 0.005`.
PASS: `P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_PASS`.
FAIL: `P_V5_R256_36FB_TWO_STYLE_OPTIMIZATION_INSUFFICIENT`.

PASS -> directly supports shared multi-asset optimization/interference as the 8×2 blocker and authorizes preregistration of a fresh 8×2 V2 certification with an evidence-backed longer schedule. It does **not** authorize unseen-family directly.

FAIL -> localize 36fb itself before any claim of shared-capacity wall, information limit, or PatchMatch need.
