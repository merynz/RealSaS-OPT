# IRIS H-GEO-P3 — Explicit High-Resolution Geometry Carrier

## Trigger

H-GEO-P2 completed on exact source `3a572989ab379975ed7cda5f0dd8b6f33de246b3` with valid apparatus and 4800 optimizer steps. P2 passed absolute distance, normal and near-surface gates and all four preregistered improvement gates versus P1, but failed the locked endpoint-p95 gate (`0.0858746 > 0.075`) and the shared-H cosine-retention gate (`0.5657004 < 0.60`).

This is sufficient evidence to open the preregistered explicit-geometry-carrier study. It is not authority to train TESSA or to declare SURF-A2 ready.

## P3 question

Can IRIS preserve the exact shared V8 evidence carrier while learning a separate, higher-spatial-frequency geometry carrier from the same frozen V8 view evidence, and can that dedicated carrier close the locked endpoint-precision court without regressing the geometry skills already demonstrated by P2?

## Architecture under test

```text
8-view prepared evidence
        ↓
frozen V8 view_bank
        ├──────────────────────────────→ frozen V8 scene + field → shared H (unchanged)
        │
        └→ dedicated geometry_scene (V8 weight copy)
             ↓
           64² geometry triplanes
             ↓
           geometry-only 2× residual refinement
             ↓
           128² geometry triplanes
             ↓
           dedicated geometry_field (V8 weight copy)
             + zero-init continuous XYZ Fourier residual
             ↓
           H_geo (384-D)
```

The shared V8 model is not in the optimizer and is not serialized into the P3 trainable checkpoint. P2 checkpoint bytes are comparison-only and forbidden for P3 initialization.

## Locked truth and training

P3 reuses the H-GEO-P1 exact authored-triangle dataset byte-for-byte. No truth rebuild or resampling is allowed. The four metric targets remain unsigned exact triangle distance, closest-surface offset, triangle normal up to sign, and near-surface classification.

Training budget is 4800 carrier steps, matching P2. The geometry scene, dedicated geometry-field body, high-resolution refiner, continuous-position residual and training-only metric head may receive gradients. Shared V8 receives none.

## Independent court

Fresh equal-capacity probes compare P3 `H_geo`, P2 H, frozen V8 H, XYZ-only, and shuffled P3 `H_geo` on the identical held-out exact-triangle queries.

P3 must pass the existing absolute geometry bar:

- mean distance MAE `<= 0.025` IRIS;
- worst-subject endpoint p95 `<= 0.075` IRIS;
- worst-subject normal abs-dot `>= 0.70`;
- worst-subject near balanced accuracy `>= 0.80`.

It must also reduce worst endpoint p95 by at least 10% versus P2, keep P2-level distance/normal/near performance within the preregistered non-regression margins, retain local-alignment superiority over XYZ/shuffled controls, and prove exact preservation of the shared V8 state and anchor H.

## Decision law

PASS authorizes design of a new high-resolution direct-surface experiment that consumes the explicit geometry carrier. PASS does not itself authorize SURF-A2 or TESSA training.

FAIL sends the program to a focused endpoint-frequency/scale audit of the explicit carrier. Threshold relaxation, teacher-in-inference, hidden repair, TESSA rescue, or P2 initialization are forbidden.
