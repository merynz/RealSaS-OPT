# RealSaS — SkinFieldCodec A0 global-temperature causal diagnostic

**Date:** 2026-09-03  
**Status:** `FALSIFIED__FREEZING_INITIAL_GLOBAL_TEMPERATURE_DOES_NOT_CLOSE_SHARP`  
**Workflow:** `33757183471`  
**Job:** `100654330332`  
**Runner:** `northcentralus`

## Question

Is co-adaptation of the single learned global Codec softmax temperature the cause of the clean ID-bound `sharp_fork_5` three-consecutive-PASS instability?

## Intervention

Two lanes used exact Bound V2 target binding and identical witness/model/seed/LR/weight-decay/horizon/acceptance:

1. `CURRENT_LEARNED_GLOBAL_TEMPERATURE` — current Codec behavior;
2. `FROZEN_INITIAL_GLOBAL_TEMPERATURE` — `log_temperature` excluded from the optimizer and held at its generic initialization value, giving temperature `1.0431472063`.

No temperature value was chosen from witness results.

## Results

### `chain_blend_3`
- learned temperature: sustained PASS step `480`, final temperature `0.986864`;
- frozen temperature: sustained PASS step `320`.

### `branch_blend_4`
- learned temperature: sustained PASS step `768`, final temperature `0.949633`;
- frozen temperature: sustained PASS step `576`.

### `sharp_fork_5`
Learned temperature:
- no sustained PASS;
- final temperature `0.883302`;
- final p95 `0.0448788`, deformation ratio `0.00790533`;
- final consecutive PASS count `1`.

Frozen temperature:
- no sustained PASS;
- final p95 `0.1180966`, deformation ratio `0.0204097`;
- final consecutive PASS count `0`.

The frozen lane still shows isolated PASS crossings but is materially worse on the sharp witness by the end of the frozen horizon.

## Conclusion

`LEARNED_GLOBAL_TEMPERATURE_IS_THE_SHARP_STABILITY_ROOT_CAUSE = FALSE`

Freezing the global temperature is not a valid source repair. It accelerates the two easier witnesses but does not close the sharp witness and worsens its final row-tail error.

Therefore:
- keep temperature learning unchanged pending stronger evidence;
- do not choose a fixed temperature from these witness traces;
- continue diagnosis at the level of objective/optimizer forces acting after a valid sharp checkpoint.

## Next diagnostic

At the first clean ID-bound scalar PASS checkpoint of `sharp_fork_5`, clone the exact model and AdamW state and apply equal 32-step continuations under isolated generic forces:
- full current objective;
- cross entropy only;
- mean pair-L1 only;
- deformation MSE only;
- reconstruction only (CE + mean L1);
- current objective with weight decay disabled after the branch point.

This diagnostic changes no frozen behavioral gate and is intended only to identify which existing training force destroys or preserves a good checkpoint.
