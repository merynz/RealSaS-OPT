# RealSaS IRIS — MapAnything Apache Orthographic Geometry V1

**Status:** implementation branch; optimizer not authorized until executable preflight PASS.  
**Shipping input:** one neutral pose × eight ordered orthographic views.  
**Purpose:** test a genuinely pretrained multi-view geometry backbone under the exact RealSaS orthographic observation contract, rather than re-training the small P-V5 learner again.

## Frozen architecture

```text
native 1024 RGBA × 8
  ├─ RGB -> pinned MapAnything Apache @ 518²
  │        DINO image encoder + exact view-rotation conditioning + multi-view transformer + DPT geometry head
  │        -> camera-frame dense geometry + confidence/ambiguity
  └─ RGBA -> native detail encoder (1024 -> 512 -> 256 -> 128)
                 ↑ coarse pretrained geometry features
                 -> dense forward depth d + geometric log-risk

known RealSaS camera:
P(v,x,y) = O(v,x,y) + d(v,x,y) F(v)
N = deterministic finite-difference normal(P)
support = deterministic raster alpha
```

Upstream predicted cameras are never authority. Product P always lies on the exact known orthographic pixel ray. No Z/descriptor head is trained in V1; correspondence descriptors may only be added after geometry qualification if common-frame P is insufficient for persistence.

## Training objective

`L = 1.0 P + 0.35 depth + 0.30 signed-normal + 0.25 same-surface multi-view consistency + 0.10 heteroscedastic risk`.

The point term is direct common-frame 3D supervision. The depth term is the exact missing orthographic DOF. Normal is derived from P, so it shapes local surface coherence without inventing a second inconsistent geometry head. Cross-view consistency uses only the existing geometry identity/XY teacher authority.

## Optimization schedule

The inference graph never changes. Stages only control optimizer access:

1. `adapter`: 1 epoch native detail/refiner warmup; pretrained base frozen.
2. `geometry`: 3 epochs; full pretrained multi-view information-sharing + DPT/scale geometry path trainable.
3. `full`: 8 epochs; image encoder also trainable at much smaller LR. The upstream pose predictor and unused depth/ray/translation encoders remain frozen. The camera-rotation encoder is active because its input is the fixed RealSaS view index, not hidden authority.

A100 + BF16 are mandatory. Gradient checkpoint flags are enabled where the pinned upstream exposes them. One family/8 views is one microbatch; gradient accumulation defaults to 4.

## Zero-step preflight gate

Before any optimizer step, `preflight.py` must PASS on the actual GPU and one authorized family. It verifies:

- exact model id/revision and branch config;
- CUDA/BF16;
- native 1024 input and eight-view order;
- target depth within configured representable range;
- raw MapAnything **forward** (never `infer`, which is inference-mode);
- exact output contract and finite loss;
- gradient reaches refiner and, for geometry/full stages, pretrained base;
- measured peak CUDA allocation;
- zero optimizer steps.

If `full` preflight OOMs on a 40 GB A100, training must not silently shrink the model or resolution. First run the `geometry` stage. Full encoder unfreeze requires a separate passing memory preflight or a larger GPU.

## Evaluation / working decision rule

Primary: matched family-disjoint `P_mean/P50/P90/P95`, family-P95 p90/worst, signed-normal median/p95. Hard witness families remain diagnostic slices, not training targets.

When a matched P-V5 baseline JSON is supplied, V1 is promoted only if:

- aggregate P95 falls by at least 20%; and
- no matched family P95 regresses by more than 10%.

This is a working architecture decision rule, not a claim that 20% is a universal product threshold. Product substrate closure remains a later SurfaceBuilder/rigging sufficiency gate.
