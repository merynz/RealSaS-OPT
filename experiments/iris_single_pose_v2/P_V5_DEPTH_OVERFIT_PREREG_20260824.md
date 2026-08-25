# IRIS Single-Pose V2 — P-V5 FIT-Only Depth Overfit Preregistration

Status: `FROZEN_PRE_OPTIMIZER`
Date: 2026-08-24

## Question

With P-V5 formulation already closed at optimizer zero, can the current IRIS V2 visual trunk plus the one-scalar camera-forward depth head fit reconstructed canonical P to the existing absolute precision target on a tiny, fixed FIT-only panel?

This is an optimization/extractability diagnostic only. It is not a generalization, matcher, product, or native-1024 qualification gate.

## Parent authority

- P ontology remains canonical/object-frame position of the observed physical surface locus.
- P-V5 execution boundary remains:

```text
native ordered 1024 RGBA
  -> derive h_native once from alpha
  -> detach/freeze h_native
  -> resize learner image to 256
  -> neural depth d
  -> analytic P using canonical yaw
```

- Teacher camera `half_extent` is forbidden as model/extractor input.
- Canonical yaw is deterministic from ordered view index V0..V7 = 0,45,...315 degrees.
- `h_native` is never recomputed after resize.

## Frozen membership

Use exactly `P_V5_DEPTH_OVERFIT_MEMBERSHIP_V1.json`:

- 8 assets;
- all are the already-frozen `FIT_TRAIN_SENTINEL` role from `P_FORMULATION_V3_PANEL_V1.json`;
- no post-result asset selection;
- no FIT_SELECT, TUNE, CAL, DEV, or EXTERNAL_HOLDOUT.

Both `cel_clean` and `ink_cel` are training/evaluation cells for each of the same 8 assets, yielding 16 asset-style cells. This is intentional within-panel overfit, not style generalization.

## Legal runtime information

Learner input:

- resized RGBA at 256;
- canonical yaw from view index;
- detached native-image-derived `h_native`.

Teacher-side supervision only:

- `primary_geometry.npz`: `vertices`, `faces`;
- `raster_authority.npz`: triangle/barycentric surface witness used to reconstruct exact canonical P at sampled visible loci.

Explicitly forbidden from the new staging/training path:

- `camera.json`;
- teacher camera half-extent;
- hidden rig/joint/owner IDs;
- weights, Pose-B mechanics, GFDR;
- persistent-track identity or correspondence targets;
- TUNE/CAL/DEV/EXTERNAL assets.

## P-only intervention

Trainable modules:

- shared image encoder;
- within-view reasoning;
- cross-view fusion;
- context fuse;
- d8/d4/d2 decoder;
- P-V5 `p_depth_head`.

Frozen / zero-authority heads:

- `n_head`;
- `u_geo_head`;
- `coarse_head`;
- `fine_head`.

No N/U/Z/matcher/reciprocal/cycle loss is active. The only optimizer objective is camera-forward depth at sampled legal P loci.

For each target P and known yaw:

```text
d_truth = P_truth dot forward(yaw)
d_pred  = P_pred  dot forward(yaw)
L_depth = SmoothL1(d_pred, d_truth; beta=0.01)
```

Loss numerics are FP32. AMP is allowed only for the neural forward on CUDA.

Evaluation authority remains full reconstructed 3D canonical P Euclidean error, not depth loss.

## Frozen optimizer protocol

- input resolution: 256;
- seed: 20260824;
- optimizer: AdamW;
- lr: 3e-4;
- betas: (0.9, 0.95);
- weight decay: 0.0;
- microbatch: 1 asset-style cell;
- 16 cells/epoch;
- epochs: 64;
- optimizer steps: exactly 1024 if run completes;
- gradient clip norm: 1.0;
- no data augmentation beyond the two frozen styles, both seen every epoch;
- candidate checkpoints: epochs 16, 32, 48, 64;
- checkpoint selection: minimum `worst_cell_P_p95`, tie -> earlier epoch.

No hyperparameter sweep or post-result rescue is authorized.

## Frozen evaluation

At every candidate checkpoint evaluate all 16 asset-style cells using the fixed cached P sample field.

Report per-cell P p50/p90/p95/max, global P p50/p90/p95/max, worst-cell P p95, best/worst asset-style identity, and optimizer steps.

## Decision

Existing absolute precision target is unchanged: `P p95 <= 0.005`.

Allowed labels:

- `P_V5_DEPTH_OVERFIT_PASS`: every one of the 16 asset-style cells has P p95 <= 0.005 at the selected checkpoint.
- `P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT`: otherwise.

A failure does not reopen `P_GEOMETRY_SUFFICIENT` or P-V5 formulation closure. It localizes the next problem to learner/optimizer/feature capacity under this intervention.

A pass authorizes designing a broader FIT/TUNE P-V5 learner gate; it does not itself authorize product/generalization claims.

## Pre-optimizer release conditions

Before optimizer step 1:

1. CPU contract/regression preflight PASS;
2. source files compile;
3. membership exactly matches frozen 8 FIT_TRAIN sentinels;
4. new staging/training path contains no `camera.json` dependency and accepts no TUNE/sealed path;
5. P-V5 native helper rejects non-1024 native input;
6. `h_native` is finite, positive, detached and transported unchanged;
7. synthetic P-V5 forward/backward finite with nonzero `p_depth_head` gradient;
8. CUDA production-width R256 AMP forward + FP32 P-only loss/backward PASS with optimizer steps = 0;
9. only the declared P-path parameters are trainable.

Any failure is fail-closed: zero scientific optimizer steps.
