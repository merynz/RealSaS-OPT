# P-V5 R256 8×2 Additional Low-LR Continuation — Result — 2026-08-25

**Scientific status:** `P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT`

This is a localization result. It does **not** rewrite the immutable parent result `P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT`.

## Frozen question

Starting from the exact failed `TAIL_0512` weights from the original 8×2 V1 run, is an additional shared low-learning-rate optimization phase sufficient to make all 16 frozen asset-style cells individually satisfy `P_p95 <= 0.005`?

Frozen intervention:
- parent checkpoint SHA-256: `0228c8c939490c9ac33cee6ca360c9228a5b4ee462b30b9774888c12129f3776`;
- same R256 architecture;
- same 8 assets × 2 styles;
- same truth and objective;
- fresh AdamW moments;
- LR `3e-5`;
- +2048 optimizer steps;
- every update represents all 16 cells through 8-cell ×2 accumulation;
- no PatchMatch;
- no camera.json;
- no new data;
- no TUNE/CAL/DEV/sealed consumption.

## Authority trajectory

| continuation step | total step incl. parent | pass cells | aggregate P95 | worst-cell P95 |
|---:|---:|---:|---:|---:|
| 0 | 2560 | 8/16 | 0.0051227102 | 0.0083204692 |
| 128 | 2688 | 8/16 | 0.0049968259 | 0.0081199796 |
| 256 | 2816 | 9/16 | 0.0048263771 | 0.0078080020 |
| 512 | 3072 | 12/16 | 0.0045912963 | 0.0073148735 |
| 1024 | 3584 | 12/16 | 0.0042027804 | 0.0065983724 |
| 1536 | 4096 | 12/16 | 0.0039720607 | 0.0061772318 |
| 2048 | 4608 | **13/16** | **0.0037062768** | **0.0057407595** |

The preregistered PASS rule required 16/16 cells at one authority checkpoint. Therefore the continuation is a scientific FAIL.

## Final blockers at CONT_2048

Only three cells remain above threshold:

| asset | style | P95 | margin above 0.005 |
|---|---|---:|---:|
| `asset_36fb02305846592b1ecdf3d4` | cel_clean | 0.0057407595 | +0.0007407595 |
| `asset_36fb02305846592b1ecdf3d4` | ink_cel | 0.0057182135 | +0.0007182135 |
| `asset_76313e4bd82b82fcd1659c70` | ink_cel | 0.0050427158 | +0.0000427158 |

`asset_76313... / cel_clean` passes at `0.0049655431`.

## Critical localization result

Comparing the parent `TAIL_0512` checkpoint to `CONT_2048`, **all 16/16 cells improve**. No cell regresses while the hard assets improve.

Examples:
- `36fb cel`: `0.0083204692 -> 0.0057407595`;
- `36fb ink`: `0.0083201882 -> 0.0057182135`;
- `76313 cel`: `0.0067907... -> 0.0049655431`;
- `76313 ink`: `0.006748... -> 0.0050427158`;
- previously failing `42512...` and `5a19...` become comfortable two-style PASSes.

The worst-cell curve is still improving at the final authority checkpoint. The `36fb` two-style mean P95 progresses approximately:

```text
CONT_0128  0.00808090
CONT_0256  0.00779549
CONT_0512  0.00728802
CONT_1024  0.00658380
CONT_1536  0.00615655
CONT_2048  0.00572949
```

The final two 512-step intervals each improve the mean by about `4.27e-4`. Therefore this experiment establishes only that **+2048 steps at the frozen low LR were not sufficient**. It does **not** establish a shared representational-capacity wall or a plateau.

A linear projection of the recent trend would cross 0.005 after roughly another ~900 steps, but that extrapolation is diagnostic only and is not a scientific result or authorization for another continuation.

## Updated evidence ranking

- style interference: **strongly disfavored**;
- one broken view / silhouette-localized hard tail: **disfavored by prior residual microscope**;
- PatchMatch-style geometry refinement as the next intervention: **not justified**;
- shared multi-asset fitting burden: **supported**;
- true shared-capacity wall: **not established and weakened by 16/16 simultaneous improvement**;
- schedule / optimization under-budget: **strongly supported**;
- intrinsic two-style extractability of `36fb...`: **still unknown**.

## Next controlled gate

Do **not** continue the failed 8×2 checkpoint again by inspection-driven extension.

Next high-information experiment:

```text
asset_36fb02305846592b1ecdf3d4
  × cel_clean
  × ink_cel
        ↓
fresh R256 model
same one-asset/two-style learner protocol
        ↓
can this asset itself reach both cells <= 0.005?
```

Interpretation:
- solo two-style PASS -> shared optimization/interference burden is directly supported; preregister a fresh 8×2 V2 certification with an evidence-backed adequate schedule;
- solo two-style FAIL -> localize `36fb` itself before any shared-capacity or PatchMatch conclusion.

Unseen-family remains closed. Architecture change remains unauthorized.
