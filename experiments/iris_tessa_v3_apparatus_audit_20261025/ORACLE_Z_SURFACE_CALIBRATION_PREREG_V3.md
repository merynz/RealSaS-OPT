# Oracle-Z Surface Calibration Court — Preregistration V3

**Date:** 2026-10-26  
**Status:** `SEALED_BEFORE_EXECUTION`

## Why V3 exists

Oracle-Z V2 repaired the optimizer/volumetric-control failure, but neither triplane nor dense-3D reached the surface-family gates. The V2 surface target mixed a nearest-component discrete attribute assignment into a continuous field, so carrier failure remained confounded with target/decoder learnability.

V3 first validates a genuinely smooth, surface-local response field with a learned direct-coordinate positive control. A carrier is interpreted only at difficulty points where the direct control solves the exact same target.

## Surface truth

Each deterministic scene contains four anisotropic ellipsoidal components. At query `x`, component `i` has normalized shell distance

`d_i = ||(x-c_i)/s_i||_2 - 1`.

Define smooth shell response

`g_i = exp(-12*d_i^2)`.

The four model-facing channels are smooth surface-local quantities:

- `y0 = sum_i g_i / 4`;
- `y1..3 = sum_i g_i * a_i / 4`, where `a_i` is a fixed authored 3-vector for the component.

No nearest-component hard attribute switch is used.

Families:
- `SURFACE_STRUCTURED`: anisotropic components with one component set to the ladder radius in all axes;
- `THIN_SURFACE`: each component has one thin axis set to the ladder radius.

Reference spacing `h_ref = 2/24`. Radius ladder: `4h_ref, 2h_ref, 1h_ref, 0.5h_ref`.

Training/evaluation points are disjoint. Each scene uses half uniform and half shell-near samples. Boundary/surface-local evaluation is the subset with `y0 >= 0.05`.

## Arms

1. `DIRECT_XYZ_PER_SCENE`
   - per-scene MLP directly on exact xyz;
   - positive control only, not a production proposal.

2. `TRIPLANE_PER_SCENE`
   - `3 x 16 x 24 x 24`, 27,648 carrier scalars;
   - bilinear XY/XZ/YZ sampling, `align_corners=False`;
   - per-scene query MLP.

3. `DENSE3D_PER_SCENE`
   - `16 x 12 x 12 x 12`, 27,648 carrier scalars;
   - trilinear sampling;
   - per-scene query MLP.

No multires/xyz diagnostic arm is reopened unless the validated surface court first demonstrates a carrier problem.

## Calibration

Independent easy scenes at radius `4h_ref`, seeds `101,211,307`, using `DIRECT_XYZ_PER_SCENE`.

Candidate learning rates: `3e-3,1e-3,3e-4`; 1400 steps. Select lowest mean held-out nRMSE, ties toward smaller LR.

Calibration PASS requires mean `<=0.10` and worst `<=0.14`.

Selected LR is frozen for all comparison arms.

## Comparison

Seeds `731,947,1217,1601,1999`.

- 2200 steps;
- train pool 4096 per scene;
- held-out 2048 per scene;
- batch 32 per scene;
- per-scene/per-channel standard-deviation-normalized MSE, scale floor 0.05.

## Firewall

Independent random train/eval regression targets under the direct-coordinate arm. Mean held-out nRMSE must be `>=0.90`.

## Interpretation gates

A radius/family point is interpretable for carrier selection only when direct XYZ has:

- mean nRMSE `<=0.12`;
- worst-seed nRMSE `<=0.18`;
- mean surface-local nRMSE `<=0.18`.

At an interpretable point, a carrier is supported if it satisfies the same three limits.

A credible triplane concern exists only when:

- direct XYZ passes;
- dense 3-D passes;
- triplane fails;
- failure repeats across seeds.

Dense 3-D is selected over triplane for V3.1 only if credible concern appears at **two adjacent ladder points in the same family** and dense improves mean nRMSE by at least `25%` relative at both points. Otherwise the lower-complexity existing triplane carrier remains the provisional pre-Truth baseline.

If direct XYZ fails, the point is not used against either carrier.

## Non-claims

This court bypasses perception, does not choose shared-scene degree, does not test shared-Q across scenes, does not choose extraction resolution, and does not authorize Truth139/FIT8/product/model promotion.
