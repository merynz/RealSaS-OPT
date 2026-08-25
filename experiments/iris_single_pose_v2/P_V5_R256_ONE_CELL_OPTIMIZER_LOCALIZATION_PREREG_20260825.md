# P-V5 R256 one-cell optimizer localization V1 — preregistration

Date: 2026-08-25

## Question

The frozen R256 one-cell gate ended at P p95 `0.005682396539486942` against a `0.005` threshold, with the best preregistered candidate at the final step 2048 and strong continued improvement from 1024→2048.

This diagnostic asks:

> Is the remaining one-cell miss explained by late-stage optimization budget / learning-rate behavior, or does the same trained representation remain unable to close the p95 tail under short controlled optimizer restarts?

## Frozen scope

Use only:

- asset `asset_76313e4bd82b82fcd1659c70`;
- style `cel_clean`;
- FIT split only;
- exact R256 architecture used by the completed one-cell gate;
- exact selected checkpoint SHA-256 `cad4421ae728848bbf9181c87e0d4912c38ec6f94d61c86cd114ad0a251c79ee`;
- exact parent selected step `2048`;
- exact parent selected P p95 `0.005682396539486942`.

No TUNE/CAL/DEV/EXTERNAL split and no hidden camera metadata may be opened.

## Zero-step checkpoint verification

Before any optimizer step:

1. Restage/cache the same frozen one-cell data using the same V5/native-scale contract.
2. Load `BEST_CHECKPOINT.pt` from the completed parent run.
3. Verify checkpoint SHA-256 exactly.
4. Verify checkpoint schema and `optimizer_steps == 2048`.
5. Evaluate the loaded model with the same evaluator.
6. Require baseline P p95 to match the persisted parent value within absolute tolerance `1e-7`.
7. Persist per-view residual metrics at optimizer step 0: p50/p90/p95/p99/max and count/fraction above 0.005.

If any verification fails, stop with `APPARATUS_OR_CHECKPOINT_DRIFT`; no optimizer arm is authorized.

## Optimizer restart arms

All arms start independently from the **same exact model state** at parent step 2048. The parent checkpoint does not contain optimizer state, therefore these are explicitly **fresh AdamW restart arms**, not exact continuation of the historical AdamW moments.

Common settings:

- objective: unchanged FP32 SmoothL1 camera-forward depth, beta `0.01`;
- evaluator: unchanged full canonical P Euclidean metrics;
- AMP neural forward: enabled;
- gradient clipping: 1.0;
- betas `(0.9, 0.95)`;
- weight decay `0`;
- no augmentation;
- all currently legal P-path parameters trainable; N/U/Z remain frozen;
- 512 optimizer steps per arm;
- evaluations at restart steps `0, 64, 128, 256, 512`.

Arms:

- `A_LR3E4_CONTROL`: lr `3e-4`;
- `B_LR1E4`: lr `1e-4`;
- `C_LR3E5`: lr `3e-5`.

Each arm resets the model to the exact same parent checkpoint before optimizer construction.

## Decision logic

For each arm, record the minimum evaluated P p95 and first restart step at which `P_p95 <= 0.005`.

Primary status:

- if any arm passes: `P_V5_R256_ONE_CELL_RECOVERY_PASS`;
- if no arm passes: `P_V5_R256_ONE_CELL_OPTIMIZER_RESTART_SWEEP_INSUFFICIENT`.

Localization label:

- if arm A passes: `BUDGET_OR_FRESH_RESTART_SUFFICIENT_AT_ORIGINAL_LR`;
- else if B and/or C passes: `LATE_STAGE_LR_FLOOR_SUPPORTED`;
- else: `SHORT_RESTART_LR_SWEEP_INSUFFICIENT`.

A PASS closes **one-cell learner/optimizer sufficiency only**. It does not establish style invariance or generalization. After PASS, the next authorized rung is still a separately preregistered `1 asset × 2 styles R256` experiment.

If all arms fail, do not reopen P ontology/V5 geometry/R256 free-field closure. Next diagnosis should inspect residual-tail structure, objective alignment, and learner feature capacity before any broader training.

## Promotion discipline

Frozen ladder remains:

```text
R256 field representation
  ↓
1 asset × 1 style learner/optimizer sufficiency
  ↓ PASS
1 asset × 2 styles
  ↓ PASS
8 assets × 2 styles
  ↓ PASS
unseen-family generalization
```
