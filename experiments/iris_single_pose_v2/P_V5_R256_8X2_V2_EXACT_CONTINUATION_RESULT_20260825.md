# IRIS P-V5 R256 — Exact 8×2 continuation PASS

**Date:** 2026-08-25  
**Status:** `P_V5_R256_8X2_V2_EXACT_CONTINUATION_PASS`

## Immutable parent
Fresh 8×2 V2 remains an immutable FAIL:
- selected `TAIL_4096`;
- total optimizer steps `6144`;
- 14/16 cells PASS;
- worst cell P95 `0.005129679851233959`.

## Exact-state continuation
The authorized continuation restored the exact V2 `TAIL_4096` model, AdamW state, GradScaler state, and Python/NumPy/Torch RNG state. No fresh optimizer moments were created. Same 16 cells, same objective, same LR `3e-5`; +1024 steps.

Zero-step parent reproduction before continuation:
- pass count `14/16`;
- worst cell P95 `0.005129679851233959`;
- max absolute per-cell P95 delta vs parent authority `0.0`.

## Result
First strict crossing:
- `EXACT_CONT_0512`, total optimizer step `6656`;
- 16/16 cells PASS;
- worst cell P95 `0.00494269179180264`.

Selected/final authority:
- `EXACT_CONT_1024`, total optimizer step `7168`;
- 16/16 cells PASS;
- aggregate P95 `0.003200733161065726`;
- worst cell P95 `0.004825880285352466`;
- selected checkpoint SHA-256 `672a92030ce1a62acd8228791eb92c7af93c34fa36ac1d5866153f38ef8708de`;
- decision SHA-256 `fa8ee6748bdeb1489d02ab7353db0d0fa6ba028f81e2537fce05709b90e0b66b`.

Hard `36fb` final:
- cel_clean `0.004782267496921113`;
- ink_cel `0.004825880285352466`.

## Scientific interpretation
This directly proves that the final V2 miss was still optimization-budget limited along the same optimizer trajectory. It is not evidence for a hard shared-capacity wall. Earlier V1/V2 FAILs remain immutable.

No PatchMatch, architecture change, camera JSON, TUNE/CAL/DEV/EXTERNAL_HOLDOUT access, or unseen-family evidence was used in obtaining this PASS.

Next: preregister a fixed-checkpoint zero-shot DEV generalization probe before opening DEV raster/geometry evidence. Preserve `EXTERNAL_HOLDOUT` sealed.
