# P-V5 R256 One-Asset / One-Style Learner Overfit V1 — Preregistration

**Date:** 2026-08-24  
**Status:** `FROZEN_BEFORE_OPTIMIZER_STEP_1`

## Question

With P ontology, P-V5 native-scale-once analytic reconstruction, and the R256 scalar depth-field representation already closed/certified, can the current visual learner optimize one fixed legal observation cell to canonical `P_p95 <= 0.005`?

This is a **fit/optimization gate**, not a generalization gate.

## Frozen membership

Exactly one pre-existing FIT sentinel and one style:

- `asset_76313e4bd82b82fcd1659c70`
- `cel_clean`

Selection was frozen before this learner run. Rationale comes only from the already-completed field-representation closure: this asset-style is the hardest tested full-R certified cell (`P_p95 ≈ 0.000817`) and is non-certified under R/2 (`≈ 0.007752`). No learner result was used to choose it.

No post-result switching is allowed.

## Frozen representation

Learner input: `8 × RGBA × 256 × 256`.

P formulation remains:

```text
P(g,theta) =
    h_native * gx * right(theta)
  - h_native * gy * up
  + d(g,theta) * forward(theta)
```

- canonical yaw is fixed by V0..V7 ordering;
- `h_native` is estimated once from native1024 RGBA and transported unchanged;
- teacher camera half extent / `camera.json` is forbidden;
- the learner predicts only camera-forward scalar depth `d`.

### Full-R requirement

The old P-V5 learner emitted depth/P on R/2=128². That representation is not certified for the frozen per-cell P threshold.

This gate therefore uses a **true R256 image-conditioned P branch**:

```text
R256 RGBA
   ├─────────────── full-resolution image stem ───────┐
   └→ encoder → ... → y2 (R128) → bilinear to R256 ──┤
                                                      ↓
                                          full-resolution fusion
                                                      ↓
                                            scalar depth d @ R256
                                                      ↓
                                             analytic P @ R256
```

A bare `R128 -> bilinear upsample -> 1×1 depth` is forbidden because it does not create a new full-resolution image-conditioned representation.

N/U/Z heads remain outside this gate and are frozen.

## Frozen objective/evaluation

Supervision uses the same deterministic visible raster-authority sampling family as prior P-V5 depth work:

- 4096 visible surface samples per view;
- 8 views;
- exact reconstructed teacher surface `P_truth`;
- fixed `pv5-depth` seed semantics.

Training objective:

`SmoothL1(d_pred, d_truth, beta=0.01)` in FP32.

Evaluation authority:

full reconstructed canonical P Euclidean error at the same fixed loci.

PASS iff:

`P_p95 <= 0.005` on the single frozen asset-style cell.

## Frozen optimization protocol

- seed: `20260824`
- optimizer: AdamW
- lr: `3e-4`
- betas: `(0.9, 0.95)`
- weight_decay: `0`
- microbatch: 1 cell (all 8 views)
- AMP: neural forward only
- objective/evaluation: FP32
- grad clip: `1.0`
- planned optimizer steps: `2048`
- candidate checkpoints: `64, 128, 256, 512, 1024, 2048`
- select minimum P_p95; ties go to earlier step
- no augmentation

Trainable P path:
- encoder
- within
- cross
- context_fuse
- d8/d4/d2
- R256 full-resolution image stem/fusion
- R256 depth head

Frozen:
- N head
- U head
- Z_coarse head
- Z_fine head

## Preflight requirements before optimizer step 1

CPU preflight must prove:
- package syntax;
- exact frozen membership;
- upstream P-V5 source byte identity;
- metadata firewall;
- R256 forward output shape;
- R256 branch receives direct full-resolution image input;
- P-only backward gives finite nonzero gradients;
- frozen heads receive no gradients;
- neural optimizer steps = 0.

GPU preflight must prove on production width R256:
- CUDA available;
- actual P output `[1,8,3,256,256]`;
- full-resolution branch activations `[8,*,256,256]`;
- finite P-only backward;
- nonzero gradient in both full-resolution image stem and depth head;
- no optimizer step;
- CUDA peak memory recorded.

The persisted parent field-closure authority must show:
- `status = P_V5_FIELD_REPRESENTATION_CLOSED`
- `full_R_certified = true`
- `current_R2_certified = false`
- `smallest_certified_field_hw = 256`
- `neural_optimizer_steps = 0`
- TUNE/sealed/camera metadata remained closed.

## Decision labels

PASS:
`P_V5_R256_ONE_CELL_OVERFIT_PASS`

FAIL:
`P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT`

A FAIL localizes to the R256 learner/optimization implementation. It does not reopen P ontology, native-scale-once geometry, or the already-certified free R256 field representation without contradictory evidence.

## Conditional next policy

Only on PASS:

`1 asset × 2 styles overfit`

No 8-asset or unseen-family learner gate is authorized by this preregistration.
