# RealSaS N-B3 — Post-Diagnostic Deterministic-N Amendment V1.1

**Date:** 2026-08-28  
**Status:** `PREREG_AMENDED_BEFORE_N_B3_OUTCOMES__PROXY27_INDEPENDENT`

Parent prereg SHA-256: `c5836e6f2bbb03aa1963601130e8a835932c84192e889d21300bb6ef560efb90`.

## Trigger

The preregistered pre-Proxy32 diagnostic showed that current X36 already makes oriented normal direction strongly decodable (pooled oriented median/p90/p95 `24.95° / 64.27° / 77.91°`). Therefore N-B3 must measure marginal explicit-N value beyond a substantial implicit orientation channel.

The same result also motivates a stronger deterministic challenger than the historical local-PCA-style `N_det(P)`. This amendment is made **before any N-B3 outcome is opened** and does not affect E0 Proxy27 qualification.

## Frozen B3 arms after amendment

```text
N0  = current X36 substrate; no explicit N feature
N3a = N0 + historical structured deterministic N(P_epsilon)
      [bridge/control to S0-B2 and parent prereg]
N3v = N0 + visibility/depth-constrained deterministic
      N_det(P_epsilon, raster, depth, support, frozen cameras)
N3b = N0 + component-masked structured deterministic N(P_epsilon)
      [oracle-component diagnostic only]
N2  = N0 + exact geometric N
      [idealized direct-N upper bound only]
```

The epsilon axis remains exactly `{0.000, 0.001, 0.003, 0.010}`. N0/N3a/N3v/N3b recompute P-dependent depth quantities from `P_epsilon`; N2 retains exact geometric N as the independent upper-bound channel while sharing the same perturbed-P substrate.

## N3v construction contract

N3v is deterministic and observation-only. It may use:

- canonical `P_epsilon`;
- frozen view/camera bases already required by E0;
- observed raster support and matched raster coordinates;
- camera-forward depth recomputed from `P_epsilon`;
- local raster/depth finite differences or robust local plane fits;
- visibility/support to orient/sign and confidence-weight per-view normal estimates;
- deterministic multi-view fusion into canonical coordinates.

It may **not** use exact mesh normals, faces/source topology as forward authority, component truth, rig truth, skin, joints, Pose B, mechanics truth, or any downstream outcome.

The deterministic field is therefore named `N_det(P, raster, depth, support, cameras)`, not merely `N(P)`.

## Interpretation

- `N3v ~= N2`: a separate learned direct-N head has little information-value justification beyond the observable substrate.
- `N3v < N2`, especially as epsilon grows: direct N is justified as a robustness/information channel not recovered by the strongest observation-derived deterministic construction.
- `N3a` remains a historical bridge; a null N3a result is not by itself evidence against N3v or N2.
- `N3b` remains diagnostic only and cannot become product authority.

No B3 result may retroactively change the frozen E0 Proxy27 qualification.
