# RealSaS P Hard-Tail Oracle — D2→D5 Localization, 2026-08-26

**Status:** `EXPLORATORY_SEQUENTIAL_DIAGNOSTIC__NOT_FORMAL_GATE`  
**Execution:** assistant-side CPU/container; no Colab.  
**IRIS optimizer steps:** `0`.  
**Heldouts:** TUNE/CAL/DEV/EXTERNAL remained closed.  
**Frozen checkpoint:** rung512 `FINAL_CHECKPOINT.pt`, global step `7168`, SHA-256 `a92fa18c46975578f2705ee3baa426cce573d4a767d42490e7599ed905f6e48e`.  
**Exact V1.4 bundle SHA-256:** `0889f6dca96ff5ac758cab2b63a31cad58a6f1765c11bdc9ee35e408b9386c01`.

## Why this ladder was run

D1 had already falsified raw RGB/alpha reprojection as a universal hard-tail fix. The requested sequence was therefore executed directly:

```text
D2 frozen learned-feature reprojection
 -> D3 exact-normal plane/patch oracle
 -> D4 oracle visibility/view selection
 -> D5 multi-scale feature consistency
```

The three witnesses remain:

- `f089`: low-texture giant-plane positive control; unusual detached quad retained as a valid observation morphology, not used as the sole representative.
- `ea593`: gauge-safe severe foreshortening/correspondence witness. Its unusual lying/horizontal orientation is **not** treated as an invalid sample or an excuse for P error; P is orientation-agnostic.
- `662ed`: gauge-safe deer/antler witness with thin structures, overlaps and repeated/cel-shaded appearance.

## Important checkpoint fact

The rung512 run was a **fresh model initialization**. `encoder`, `within`, `cross`, `context_fuse`, decoder stages, `p_s1_stem`, `p_s1_fuse`, and `p_depth_head` were trained. The auxiliary `n_head`, `u_geo_head`, `fine_head`, and `coarse_head` were not trainable in this run. Therefore D2 does **not** pretend that those frozen-random heads are learned descriptors. It tests only genuinely trained feature tensors from the P network.

## D2 — frozen learned-feature reprojection

Same candidate-depth assay as D1: 321 forward-depth hypotheses over the gauge-safe range; V0/V2/V4/V6 references; 1400 loci per reference (`5600/asset`); known-camera projection into the seven other views. Teacher geometry is evaluation-only and teacher visibility is absent from scoring.

Representations tested: trained `encoder.s2`, `p_s1_stem`, and `p_s1_fuse/_P_FULLRES_FEATURE`, with cosine feature cost plus image-only foreground veto.

| asset | matched direct P95 | best D2 P95 | best D2 representation / aggregation | verdict |
|---|---:|---:|---|---|
| f089 | 0.416423 | **0.123106** | `enc_s2 / mean7` | strong positive control |
| ea593 | **0.471771** | 0.574262 | `enc_s2 / median7` | representative FAIL |
| 662ed | **0.271590** | 0.304407 | `p_stem / median7` | representative FAIL |

For f089, `encoder.s2 / mean7` reaches P50 `0.002924`, P90 `0.077803`, P95 `0.123106`. Thus the exact same frozen representation can be highly discriminative on the positive-control morphology.

For ea593 and 662ed, every tested current feature representation remains worse than the direct P predictor at P95. This falsifies **current frozen feature cosine + candidate reprojection** as the general solution.

### Direct failure is strongly view-structured

Matched D2 direct P95 by cardinal reference:

| asset | V0 | V2 | V4 | V6 |
|---|---:|---:|---:|---:|
| ea593 | 0.649064 | 0.062547 | 0.618926 | 0.065106 |
| 662ed | 0.339891 | 0.054731 | 0.287579 | 0.057862 |

The representative tail is therefore not uniform noise: V0/V4 are dramatically harder than V2/V6 on both assets. This is valid morphology-dependent observability/correspondence stress, not a semantic orientation exception.

## D3 — exact-normal plane-warp oracle

Because the current rung512 `n_head` was not trained, using its output would not test the intended hypothesis. D3 instead gives the mechanism its strongest fair diagnostic: **exact teacher face normal** solely for the local plane warp. Teacher visibility remains absent.

Hard references V0/V4 only, 400 loci/view, 3×3 feature patch, `encoder.s2` evidence.

| asset | matched direct P95 | point-feature median7 P95 | exact-normal plane median7 P95 |
|---|---:|---:|---:|
| ea593 | 0.676354 | 0.685487 | **0.656420** |
| 662ed | **0.329607** | 0.362542 | 0.358451 |

Exact normal gives a modest local improvement on ea593 and essentially no closure on 662ed. **Surface orientation / plane warp is not the missing general mechanism.**

## D4 — oracle visibility/view-selection after D2

D4 uses frozen `encoder.s2` scoring and grants teacher visibility only to choose which cardinal target views are visible at truth. This is deliberately an oracle: if view selection were the causal missing piece, it should rescue the representative tail.

| asset | baseline learned-feature P95 | teacher-visible oracle P95 | zero cardinal-visible fraction |
|---|---:|---:|---:|
| f089 | 0.391596 | 0.815709 | 0.126 |
| ea593 | 0.701307 | 0.689757 | 0.519 |
| 662ed | 0.370358 | 0.412018 | 0.291 |

Ea593 improves only slightly and remains catastrophic; f089 and 662ed get worse under this restricted oracle. **Occlusion/view selection is not the sole missing mechanism and cannot justify PatchMatch by itself.**

## D5 — multi-scale geometric consistency

An ACMM-like frozen hierarchy was tested using only trained shared encoder features:

```text
s8 @ 32px: retain top64 depths
 -> s4 @ 64px: retain top16
 -> s2 @ 128px: choose final depth
```

No teacher evidence enters scoring.

| asset | direct P95 | single s2 best shown P95 | hierarchy P95 (mean) | truth retained s8 top64 | truth retained after s4 top16 |
|---|---:|---:|---:|---:|---:|
| f089 | 0.418936 | 0.130134 | 0.132531 | 0.991 | 0.828 |
| ea593 | **0.472422** | 0.576046 | 0.602338 | 0.618 | **0.253** |
| 662ed | **0.267765** | 0.328551 | 0.343820 | 0.840 | **0.338** |

The hierarchy does not close either representative asset. Critically, ea593 retains truth in coarse top64 only ~62% and only ~25% after the intermediate top16; 662ed drops from ~84% to ~34%. Search/propagation cannot recover candidates that the evidence has already ranked away.

## D2b — small frozen-feature metric probe

A deliberately lightweight follow-up tested whether the feature tensor might contain useful correspondence information that simple cosine fails to expose. Frozen `encoder.s2` pair descriptors (`|f1-f2|`, product, yaw terms) were fed to a standardized logistic classifier. For each leave-one-asset-out fold, the classifier trained on exact visible correspondences + epipolar hard negatives from the other two FIT witnesses. IRIS remained frozen; this classifier training is not an IRIS optimizer step.

| held-out asset | matched direct P95 | learned pair-metric P95 | pair-metric top8 |
|---|---:|---:|---:|
| f089 | 0.414225 | 0.397693 | 0.271 |
| ea593 | **0.464965** | 0.798937 | 0.158 |
| 662ed | **0.275291** | 0.410630 | 0.167 |

This tiny 3-asset LOAO metric probe is **negative on ea593 and 662ed** and only marginally changes f089. It is too small to prove the frozen representation lacks correspondence information, but it falsifies the easiest claim that a trivial pairwise metric on these features immediately solves the tail.

## Localization decision

What has survived:

```text
known cameras
+ candidate common-frame P
+ cross-view verification as an operation
```

What did **not** survive as a general representative-tail solution:

```text
raw RGB / alpha matching                         [D1]
current frozen feature cosine matching           [D2]
+ exact normal / local plane warp                [D3]
+ oracle visibility / source-view selection      [D4]
+ simple s8->s4->s2 coarse-to-fine pruning       [D5]
+ trivial 3-asset pairwise metric probe          [D2b]
```

The strongest current localization is therefore:

> The representative gauge-safe tail is bottlenecked **before or at cross-view correspondence evidence**. The current P features/metric do not rank the correct surface correspondence reliably under severe foreshortening, thin/overlapping structures, repeated colors and view-dependent cel appearance. Search machinery cannot manufacture a discriminative score after truth has been pruned.

This does **not** prove an information limit, and it does **not** yet prove that the encoder itself must change. D2b is too small for that inference.

## Next executable experiment

`BROADER FIT-DISJOINT CROSS-VIEW CORRESPONDENCE PROBE — METRIC VS REPRESENTATION`

Use many other open FIT families for teacher-derived positive correspondences and hard epipolar negatives; keep ea593/662ed as frozen held-out witnesses. Compare:

1. frozen encoder feature + learned correspondence metric;
2. if that fails, correspondence-aware feature adaptation/training;
3. only after correspondence ranking is proven should PatchMatch-style propagation/search be reconsidered.

## Authorization

`D2 = COMPLETE_NEGATIVE_ON_REPRESENTATIVE_TAILS__POSITIVE_F089_CONTROL`  
`D3 = COMPLETE_NO_CLOSURE`  
`D4 = COMPLETE_NO_CLOSURE`  
`D5 = COMPLETE_NO_CLOSURE`  
`D2b_3ASSET_LOAO = EXPLORATORY_NEGATIVE_REPRESENTATIVE`  
`FULL_PATCHMATCH = NOT_AUTHORIZED`  
`LONGER_P_TRAINING = NOT_NEXT`  
`CAL_DEV_EXTERNAL = CLOSED`.
