# RealSaS N-B3 — Explicit-N Marginal-Value + Epsilon Prereg V1

**Date:** 2026-08-28  
**Status:** `PREREGISTERED_BEFORE_N_B3_OUTCOMES__PROXY32_INDEPENDENT`

## Question

Does an explicit/direct normal channel add downstream information beyond exact/noisy P plus the orientation information already implicit in the current support/raster/depth substrate?

The baseline is not called "zero-N information": the current 36D slot may implicitly encode orientation. The pre-Proxy32 implicit-N probe quantifies that decodability.

## Arms

```text
N0  = current 36D substrate; no explicit N feature
N3a = N0 + structured deterministic N(P_epsilon), observation-only
N3b = N0 + component-masked structured N(P_epsilon), ORACLE component diagnostic only
N2  = N0 + exact geometric N, idealized direct-N upper bound
```

N3b is diagnostic only and may not become product authority. N2 exact N is target/upper-bound authority only.

## Controlled P-noise axis

Use deterministic zero-mean isotropic Gaussian P perturbation in canonical object coordinates with per-point seed namespaced by asset, view/carrier and epsilon. Frozen absolute standard deviations:

```text
epsilon in {0.000, 0.001, 0.003, 0.010}
```

For N0/N3a/N3b, P xyz and all deterministic camera-forward depth fields are recomputed from P_epsilon. Raster XY and observed support remain observation provenance. N3a/N3b normals are recomputed from the perturbed P field; they may not reuse clean-P normals. N2 keeps exact geometric N as the idealized independent direct-head upper bound while sharing the same P_epsilon substrate.

## Interpretation

- N3a ~= N2 across epsilon: direct N head has little information-value justification.
- N3a < N3b ~= N2: component/sheet separation is the bottleneck, not direct-N prediction.
- N3a ~= N3b < N2 and gap grows with epsilon: direct N is justified primarily as a P-noise-robust channel, not because downstream consumers intrinsically require normals.
- Any N0 result is interpreted relative to an implicit-orientation baseline quantified separately.

No B3 outcome may retroactively change the E0 Proxy32 qualification margins.
