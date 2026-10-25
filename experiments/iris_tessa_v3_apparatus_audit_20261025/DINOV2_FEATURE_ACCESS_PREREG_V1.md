# RealSaS — Exact DINOv2-S/14 Feature-Access Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** frozen-foundation CPU probe; no product/ModelRelease authority.

## Authorization

The preregistered post-foundation IRIS subcell-routing court passed at 1.0 bit accuracy on all five seeds with a clean random-label firewall. Therefore the next bounded question is upstream: whether the current frozen foundation representation preserves graphics-relevant local RGB detail before the shared scene sees it.

The RGBA collision proof remains separate. This court cannot repair or test information already destroyed by RGBA -> white-composited RGB.

## Exact current authority

Use the same authority sealed by `models/iris/v2/dinov2_foundation_v2.py`:

- repository: `facebookresearch/dinov2`
- source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- constructor: `dinov2_vits14`
- weight SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- weight size: `88283115`
- native source input: RGB 1024x1024
- exact sealed preprocess: bicubic+antialias 1024 -> 518, ImageNet normalization
- patch grid: 37x37
- embed dim: 384
- 12 transformer blocks.

No foundation parameter is updated.

## Scientific question

At fixed backbone, weights, input resolution, patch grid and probe capacity, is local subpatch information more linearly accessible in an earlier/middle transformer layer than in the current final normalized patch token?

A positive answer means the current `level_ids=(x_norm_patchtokens,)` feature contract may be the bottleneck; it does not justify replacing DINOv2.

## Synthetic RGB task

Generate native 1024x1024 RGB canvases with neutral gray background.

Use the central 7x7 cells of the exact logical 37x37 patch partition. For every active logical cell independently sample four fair bits. Each bit controls a small high-contrast square centered in one quadrant of the logical cell.

Four native-square side-length strata are fixed before outputs:

- 2 px
- 4 px
- 8 px
- 12 px

The logical cell boundaries are defined by integer-rounded partitions of `[0,1024)` into 37 bins. Square placement is defined in native coordinates before the exact sealed resize.

Each canvas independently randomizes all active-cell bits. This removes semantic, smoothness and neighboring-cell shortcuts.

## Dataset

For each width stratum:

- 12 train canvases
- 6 evaluation canvases
- fixed generation seeds derived from master seed `240139` and width; train/eval streams disjoint.

Each canvas contributes 49 patch-local samples, each with four binary targets. Thus probe training uses 588 local samples per width and evaluation uses 294.

The foundation sees each canvas once per extraction; all compared layers are captured from the same forward inputs.

## Feature arms

DINOv2-S has 12 blocks indexed 0..11.

Extract normalized patch tokens from exact block indices:

- `L2`: block index 2
- `L5`: block index 5
- `L8`: block index 8
- `L11`: block index 11 (same final block depth as current output)

Also extract current `FINAL_CONTRACT = forward_features()['x_norm_patchtokens']`.

Expected identity check: `L11` and `FINAL_CONTRACT` must be numerically equal within max absolute error `1e-6`; otherwise the court is invalid.

Each arm is 384-D. No concatenation is used in V1, avoiding dimensionality/capacity confounding.

## Readout

For every width x feature arm fit the same independent linear probe:

`Linear(384,4)`

using BCE-with-logits on the frozen feature rows.

- AdamW
- lr `3e-3`
- betas `(0.9,0.999)`
- weight decay 0
- 800 full-batch optimization steps
- initialization seed `731`
- no scheduler, clipping, early stopping, hyperparameter search, augmentation or outcome-dependent tuning.

Because feature extraction is deterministic and the readout task is tiny, V1 uses one sealed readout seed; statistical variation is across held-out canvases and width strata rather than repeated backbone training.

## Metrics

Per width x arm:

- held-out bit accuracy over all four bits;
- held-out BCE;
- per-bit accuracy;
- feature variance diagnostics.

Aggregate:

- mean accuracy across all four width strata;
- minimum width-stratum accuracy;
- paired accuracy delta against `FINAL_CONTRACT` per width.

## Controls / validity

- target positive fraction per bit must lie in `[0.40,0.60]` on train and eval;
- `L11` vs `FINAL_CONTRACT` max abs error <= `1e-6`;
- all feature tensors finite and expected shape `[N,1369,384]`;
- no gradients on foundation parameters;
- exact source/weight authority verification must PASS.

If any condition fails: `DINOV2_FEATURE_ACCESS_COURT_INVALID`.

## Preregistered interpretation

Let `A_final(w)` be FINAL_CONTRACT held-out bit accuracy at width `w`, and `A_best_earlier(w)` the best of L2/L5/L8.

### CURRENT_FINAL_ACCESS_SUFFICIENT_TINY_V1

Call when:

- FINAL_CONTRACT accuracy >= `0.95` for every width;
- and no earlier arm improves by >= `0.03` on any width.

This closes the synthetic local-RGB-access concern at this scale; it does not prove real-image sufficiency.

### EARLIER_FEATURE_ACCESS_MATERIALLY_BETTER_TINY_V1

Call when:

- some earlier arm beats FINAL_CONTRACT by >= `0.05` on at least two width strata;
- mean earlier-vs-final paired gain for that same layer across widths >= `0.03`.

This authorizes a separately preregistered same-backbone feature-contract treatment (e.g. earlier/multilevel access) before any DINOv3 comparison.

### CURRENT_DINOV2_LOCAL_RGB_ACCESS_NOT_ESTABLISHED_TINY_V1

Call when:

- FINAL_CONTRACT fails 0.90 on any width;
- and no single earlier layer satisfies every-width >=0.95.

This authorizes a broader foundation treatment court including input resolution and alternative backbones such as DINOv3.

Otherwise: `DINOV2_FEATURE_ACCESS_INCONCLUSIVE_TINY_V1`.

## What no outcome authorizes

No outcome authorizes:

- changing current product foundation;
- DINOv3 adoption;
- Truth139/FIT8 training;
- ModelRelease promotion;
- product activation;
- a full shared-scene hypothesis PASS.
