# RealSaS — DINOv2 Dual-Level View-Token Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** frozen-foundation CPU feature-contract treatment; no product/ModelRelease authority.

## Authorization

The exact DINOv2-S/14 feature-access court established that L2 contains materially more linearly accessible local/subpatch information than the current final-only contract, while the post-foundation scene route can carry distinct subcell states when supplied.

This authorizes a same-backbone feature-contract treatment before any DINOv3/backbone replacement.

## Question

Can IRIS retain the current final DINO representation while adding early local detail **without increasing patch-token count, scene width, or image-projection parameter count**?

## Exact foundation/data

Reuse the exact authority, preprocessing, synthetic canvases and feature extraction from `DINOV2_FEATURE_ACCESS_PREREG_V1.md`:

- exact DINOv2-S/14 source/weights;
- native RGB 1024 -> exact 518 preprocess;
- central 7x7 patch-local four-bit task;
- square widths 2/4/8/12 px;
- L2 and FINAL_CONTRACT 384-D patch features;
- 12 train / 6 eval canvases per width.

Foundation remains frozen.

## Scene-token adapter arms

All arms output exactly one 512-D image-memory contribution per patch. Camera/XY are intentionally omitted from this feature-retention court because they are identical external additive signals across arms and contain neither local payload nor final-feature target.

### A — CURRENT_FINAL_512

`Linear(384 final -> 512)`

This matches current `CameraAwareViewBankVNextV1.image_projection` shape.

Adapter trainable parameters: exactly `384*512 + 512 = 197120`.

### B — EARLY_L2_512

`Linear(384 L2 -> 512)`

Same adapter parameter count as A. Diagnostic arm: tests the tradeoff if final features are discarded.

### C — DUAL_L2_FINAL_MATCHED_512

Two branches concatenated:

- `Linear(384 final -> 384)`
- `Linear(384 L2 -> 128)`

Concatenated output dim = 512.

Total adapter parameters:

`384*384+384 + 384*128+128 = 197120`.

Thus C exactly matches current image-projection parameter count and leaves patch-token count / scene-memory sequence length unchanged.

## Shared readouts / objective

Every arm receives the same two trainable readouts from its 512-D adapter output:

1. `local_head: Linear(512,4)` -> four local bit logits;
2. `final_reconstruction_head: Linear(512,384)` -> reconstruct the frozen current FINAL_CONTRACT feature.

Joint loss:

`L = BCE(local_bits) + (1 - cosine_similarity(pred_final, target_final)).mean()`

No arm-specific weights.

The final-reconstruction term is a direct representation-retention probe, not a claim that future IRIS training should distill final DINO features with this loss.

## Training

Combine all four width strata into one training set.

Seeds exactly `731, 947, 1217, 1597, 2017`.

For each seed/arm:

- deterministic initialization;
- AdamW lr `1e-3`, betas `(0.9,0.999)`, weight decay 0;
- exactly `1200` full-batch steps;
- no schedule, clipping, early stopping or tuning.

Foundation features are extracted once and detached.

## Metrics

Per seed and width:

- local four-bit accuracy;
- final-feature cosine similarity.

Aggregate per arm:

- mean/min local accuracy across seed x width;
- mean/min final cosine across seed x width;
- paired C-A local gain per seed;
- paired C-A final-cosine delta per seed;
- adapter parameter counts.

## Validity controls

- exact foundation authority must pass;
- L11 identity to FINAL_CONTRACT remains exact as in prior court;
- all features finite;
- all adapter parameter counts A/B/C must equal `197120`;
- current A must reach mean final cosine >= `0.98`; otherwise the retention readout/objective is invalid.

Failure -> `DUAL_LEVEL_VIEW_TOKEN_COURT_INVALID`.

## Preregistered positive result

Call `DUAL_LEVEL_VIEW_TOKEN_SUPPORTED_TINY_V1` when all validity controls pass and C satisfies:

1. local accuracy >= `0.98` for every seed x width;
2. mean local accuracy >= `0.995`;
3. final cosine >= `0.98` for every seed x width;
4. C mean final cosine is no more than `0.01` below A;
5. C beats A mean local accuracy by >= `0.05` in at least 4/5 seeds;
6. adapter parameter counts exactly match.

If C preserves final representation but local thresholds are weaker: `DUAL_LEVEL_RETENTION_OK_LOCAL_GAIN_INCONCLUSIVE_TINY_V1`.

If C gains local information but materially loses final representation: `DUAL_LEVEL_LOCAL_GAIN_FINAL_RETENTION_FAIL_TINY_V1`.

Otherwise: `DUAL_LEVEL_VIEW_TOKEN_NOT_SUPPORTED_TINY_V1`.

## Consequence

A positive result authorizes a separately preregistered current-scene-core treatment using the matched dual-level image contribution before considering DINOv3.

No outcome authorizes product adoption, Truth139/FIT8 training, ModelRelease promotion or a claim that DINOv3 is unnecessary in all future stages.
