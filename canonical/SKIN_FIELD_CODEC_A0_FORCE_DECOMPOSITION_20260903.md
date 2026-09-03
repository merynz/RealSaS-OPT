# RealSaS — SkinFieldCodec A0 post-PASS force decomposition

**Date:** 2026-09-03  
**Status:** `DEFORMATION_TERM_CAUSALLY_DESTABILIZES_GOOD_SHARP_CHECKPOINT__FULL_PANEL_ABLATION_REQUIRED`  
**Workflow:** `33757516425`  
**Job:** `100655405493`  
**Runner:** `eastus2`

## Question

Once the correctly bound `sharp_fork_5` Codec has already entered the frozen scalar PASS region, which existing training force first drives it back out?

## Branch point

The diagnostic trained the unchanged current A0 objective until the first individual scalar PASS, then cloned the exact Codec model state and AdamW optimizer state.

Branch checkpoint:
- step `1280`;
- row-L1 p95 `0.0453200787`;
- deformation ratio `0.0090788761`;
- simplex residual `1.19e-7`;
- negative weights `0`.

Every continuation used the same starting model and, except where explicitly stated, the same loaded AdamW state and same `lr=1e-3`, `weight_decay=1e-4`. Each branch ran exactly 32 further steps.

## Results at continuation step 32 / absolute step 1312

### Current objective = reconstruction + deformation MSE
- p95 `0.0831804127`;
- deformation ratio `0.0204366706`;
- `FAIL`.

### Reconstruction only = CE + mean pair-L1
- p95 `0.0495637693`;
- deformation ratio `0.0080801426`;
- `PASS`.

### Cross entropy only
- p95 `0.0590760671`;
- deformation ratio `0.0100516686`;
- `FAIL`.

### Mean pair-L1 only
- p95 `0.0862907171`;
- deformation ratio `0.0159394313`;
- `FAIL`.

### Deformation MSE only
- p95 `0.789708793`;
- deformation ratio `0.162961543`;
- `FAIL`.

This isolated lane inherits Adam moments from the pre-branch mixed-objective history and therefore must not be interpreted as a standalone statement about fresh deformation-only optimization. It is retained as telemetry only.

### Current objective with weight decay disabled after branch
- p95 `0.0831786394`;
- deformation ratio `0.0204368997`;
- `FAIL`.

This is essentially identical to current behavior.

## Causal conclusions

### Weight decay

`WEIGHT_DECAY_CAUSES_THE_POST_PASS_EXIT = FALSE`

Disabling weight decay after the exact branch point does not materially change the 32-step trajectory.

### Deformation term in the current mixed objective

The cleanest paired intervention is:

`current = reconstruction + deformation_mse`

versus

`reconstruction_only = reconstruction`.

Both inherit the exact same good model and Adam state. Removing only the deformation-MSE contribution changes the 32-step result from FAIL to PASS:
- p95 improves from `0.08318` to `0.04956`;
- deformation ratio also improves from `0.02044` to `0.00808`.

Thus, in the current co-optimized state, adding the deformation-MSE gradient is causally destabilizing rather than protective. It degrades both the dense-W acceptance metric and the very deformation consequence it is intended to improve.

This is evidence against keeping the current deformation term by default in Codec A0 training, but one witness/one branch is not sufficient source-repair authority.

## Required next gate before source repair

Run a full ID-bound A0 ablation on all three frozen witnesses under identical seeds/model/LR/WD/horizon/evaluation/stability:
1. current A0 objective;
2. reconstruction-only A0 objective.

Deformation remains an authoritative evaluation criterion in both lanes. No threshold, seed, horizon or three-consecutive-PASS rule changes.

Only if reconstruction-only closes the heterogeneous panel generically should A0 source be changed. A1 must remain untouched until A0 source behavior is closed.
