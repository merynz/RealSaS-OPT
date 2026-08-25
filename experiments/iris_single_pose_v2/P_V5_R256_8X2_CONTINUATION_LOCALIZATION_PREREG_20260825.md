# P-V5 R256 — 8×2 Additional Low-LR Optimization Localization V1

**Date:** 2026-08-25  
**Status:** `PREREGISTERED__NOT_RUN`  
**Scientific role:** localization only; the original 8×2 result remains immutable `P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT`.

## Why this gate exists
The frozen 8×2 V1 run failed 16/16 certification at `TAIL_0512` but continued to improve through the final authority checkpoint. Residual microscopy on the worst asset (`36fb...`) and on the prior solo-PASS control asset (`76313...`) did not localize the dominant degradation to silhouette/occlusion boundaries or one bad view. The learned camera-forward depth component degrades broadly under the shared 8-asset fit, while analytic screen-plane geometry remains effectively unchanged.

This gate asks one narrower question before any architecture intervention:

> Starting from the failed 8×2 `TAIL_0512` model weights, is additional shared low-LR optimization sufficient to bring all 16 frozen cells below the canonical P threshold?

It does **not** ask unseen-family generalization and it does **not** retroactively change the original 8×2 FAIL.

## Frozen parent
- parent run: `IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_JOINT_FIT_V1`
- parent scientific status: `P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT`
- parent selected checkpoint: `TAIL_0512`
- parent total optimizer steps: `2560`
- parent checkpoint SHA-256: `0228c8c939490c9ac33cee6ca360c9228a5b4ee462b30b9774888c12129f3776`

## Frozen membership / truth / architecture
Unchanged from 8×2 V1:
- same eight frozen FIT assets × `cel_clean`,`ink_cel` = 16 cells;
- same 4096 deterministic visible truth loci per view;
- same R256 architecture;
- same P/depth objective only; N/U/Z frozen;
- same 8-cell ×2 gradient accumulation, so every optimizer update represents all 16 cells;
- no augmentation, TUNE/CAL/DEV/EXTERNAL, sealed family, `camera.json`, teacher camera half-extent, PatchMatch or architecture change.

The runner must reconstruct the same cached truth and verify the eight parent truth SHA-256 values before optimizer step 1.

## Frozen continuation optimizer
Because the parent checkpoint contains model weights but not optimizer state, this experiment cannot isolate *pure extra-step budget* from *optimizer reset*. Its admissible claim is therefore deliberately narrower: **additional low-LR optimization sufficiency with fresh AdamW moments**.

- initialization: exact parent `TAIL_0512` model state;
- AdamW fresh moments;
- LR `3e-5`;
- betas `(0.9,0.95)`;
- weight decay `0`;
- continuation steps: `2048`;
- authority evals: `CONT_INIT`, then continuation steps `128,256,512,1024,1536,2048`;
- AMP and gradient clipping remain identical to V1.

No LR search, early stopping, adaptive schedule or result-driven checkpoint insertion is permitted.

## PASS / FAIL
At one preregistered continuation checkpoint, all 16 cells separately require `P_p95 <= 0.005`.

**PASS:** `P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_SUFFICIENT`. This supports the claim that further shared low-LR optimization from the failed weights is sufficient. It does **not** certify the original 8×2 gate and does not authorize unseen-family. Next: preregister a fresh 8×2 V2 certification run with an evidence-backed adequate schedule.

**FAIL:** `P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT`. Shared capacity/interference becomes more plausible. Remain at 8×2 for capacity/interference localization before architecture change.

PatchMatch-style geometry-in-the-loop remains a preserved intervention candidate, but is **not justified by the current broad residual pattern and is forbidden in this gate**.
