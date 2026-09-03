# RealSaS — SkinFieldCodec A0 hard-tail causal diagnostic

**Date:** 2026-09-03  
**Status:** `FALSIFIED__TOP10_TAIL_WEIGHT_1_DOES_NOT_CLOSE_SHARP_STABILITY`  
**Workflow:** `33756798991`  
**Job:** `100653091683`  
**Runner:** `westus`

## Question

Does adding the already-generic A1 top-fraction row-L1 tail objective to ID-bound Codec A0 close the clean `sharp_fork_5` three-consecutive-PASS stability seam without changing the frozen behavioral panel?

## Frozen diagnostic conditions

Both lanes used:
- exact Bound V2 canonical surface-ID binding;
- same witness definitions and seeds;
- same Codec architecture;
- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- same 1536-step horizon;
- evaluation every 32 steps;
- same A0 thresholds;
- same requirement for three consecutive PASS checks.

Lanes:
1. `CURRENT_A0_OBJECTIVE`;
2. `CURRENT_PLUS_TOP10_ROW_L1_TAIL`, `fraction=0.10`, `weight=1.0`.

No product threshold, seed, horizon or behavioral gate was changed.

## Results

### `chain_blend_3`
- current objective: sustained PASS step `480`;
- +tail: sustained PASS step `1056`.

### `branch_blend_4`
- current objective: sustained PASS step `768`;
- +tail: sustained PASS step `1376`.

### `sharp_fork_5`
Current objective:
- no sustained PASS;
- final step `1536`: p95 `0.0448787585`, deformation ratio `0.0079053342`;
- final consecutive PASS count `1`.

+top10 tail:
- no sustained PASS;
- final step `1536`: p95 `0.0899341777`, deformation ratio `0.0125349201`;
- final consecutive PASS count `0`;
- isolated threshold PASSes still occur, including step `1216` p95 `0.0493978` and step `1504` p95 `0.0400341`.

## Conclusion

`TOP10_ROW_L1_TAIL_WEIGHT_1_CLOSES_SHARP_A0_STABILITY = FALSE`

The intervention does not stabilize the clean sharp witness and materially delays sustained PASS for the two easier witnesses.

Therefore:
- do not add this tail configuration to A0 source;
- do not tune the tail weight/fraction against `sharp_fork_5` after observing this result;
- keep `top_fraction_row_l1_tail_v1` in its existing generic A1 role;
- continue diagnosis of the clean A0 stability seam without changing the frozen behavioral panel.

## Next causal question

The Codec has one learned global softmax temperature shared across every surface row and joint:

`temperature = softplus(log_temperature) + temperature_floor`.

A diagnostic will compare the current co-adaptive temperature lane with a lane where `log_temperature` is excluded from optimization and remains at its generic initialization value. All other conditions remain unchanged.
