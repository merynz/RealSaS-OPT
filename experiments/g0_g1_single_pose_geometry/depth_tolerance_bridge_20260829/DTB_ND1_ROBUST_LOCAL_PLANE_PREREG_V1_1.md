# DTB-ND1 — Robust Local-Plane Differential-Normal Replay Preregistration V1.1

**Date:** 2026-08-29  
**Status:** `FROZEN_BEFORE_DTB_ND1_OUTCOMES`

This V1.1 supersedes V1 before any DTB-ND1 scientific outcome was opened.

## Parent authorities

- `DTB-S1` = sealed 65-cell structured depth surface/substrate grid (artifact filenames retain `M4_*`).
- `DTB-S1R` = sealed ALL8 boundary refinement (artifact filenames retain `M4R_*`).
- Parent DTB-S1 canonical full result SHA-256: `9de6c8aa150cda3b333753dd58359d93bd688271e0c9e5d7aa2182b2030d90fb`.
- Parent DTB-S1R result SHA-256: `fb155acdeffaf914071c9aa75c65cbaed5eac7aab43afe135fca633cfd53c116`.

## Question

Is the current DTB-S1R ALL8 tolerance boundary primarily limited by the historical stride-2 tangent/cross-product `N_d` operator rather than by depth-position error itself?

## Single intervention

Replace only the differential-normal derivation used by the persistence matcher on clean and corrupted observable P with a deterministic robust local-plane operator.

Everything else remains frozen:

- exact same 8 `calibration_anchor8` families;
- exact same ray-depth corruption generator and seeds;
- exact same source-anchor sampler;
- exact same BASE matcher thresholds;
- exact same reciprocal `cycle_P <= 0.003` D2 admission;
- exact same D2 Geppetto/Arachne proxy checkpoints and target packs;
- exact same six proxy decision checks/margins;
- exact same typed Compiler surface adapter;
- training steps = 0;
- Proxy27 not used for tuning;
- DEV32 closed.

## Robust local-plane operator — frozen parameters

Native raster resolution remains 1024.

For each requested visible raster row:

1. Candidate image-grid offsets are the fixed 7×7 stride-2 lattice:
   `dx,dy ∈ {-6,-4,-2,0,2,4,6}`, excluding `(0,0)`.
2. Missing raster rows are ignored; no interpolation across absent/background pixels.
3. Candidate spatial weight:
   `w_img = exp(-(dx^2+dy^2)/(2*3^2))`.
4. Compute 3D center-to-neighbor distances on the observable P field.
5. Robust distance gate is computed per center from valid candidates:
   - `m = median(distance)`;
   - `mad = median(|distance-m|)`;
   - `sigma_r = max(1.4826*mad, 0.25*m, 1e-6)`;
   - retain candidates with `distance <= m + 3*sigma_r`.
6. At least 6 retained neighbors are required; otherwise `N_valid=false`.
7. Fit a weighted local plane to center + retained neighbors using weighted covariance about the weighted centroid.
8. `N_d` is the unit eigenvector of the smallest covariance eigenvalue.
9. Normal sign is non-authoritative and all current persistence continuity checks remain sign-invariant.
10. No historical-normal fallback is permitted for an invalid local-plane fit.

The implementation may compute normals sparsely/on-demand for rows actually queried by the frozen matcher; sparse execution is an optimization only and must return the same operator defined above.

These parameters are frozen before DTB-ND1 outcomes are inspected. No radius/sigma/gate tuning is allowed after the first outcome.

## Replay grid

Primary replay is the exact DTB-S1R ALL8 grid:

- zero baseline;
- epsilon RMS: `0.00175, 0.00200, 0.00225, 0.00250, 0.00275`;
- ell: `0,4,16,64`;
- asymmetry: `ALL8` only;
- 21 cells total.

Additionally, mechanism witness cells from DTB-S1 may be replayed without decision authority:

- `epsilon=.006, TWOOPP, ell=0/4/16/64`.

## Clean non-inferiority gate

Before any tolerance widening can be interpreted, the robust-operator zero-corruption baseline must be compared to the sealed historical DTB-S1R zero baseline.

Use the exact same six frozen practical-margin checks, with:

- candidate = robust local-plane zero baseline;
- reference = historical DTB-S1R zero baseline.

If the robust zero baseline fails any clean non-inferiority check, the operator is **not promotable** as a cheaper tolerance lever in this gate, regardless of its noisy-cell relative tolerance.

This clean gate is evaluated before interpreting the robust tolerance bracket.

## Tolerance decision operator

If and only if clean non-inferiority passes:

- the robust zero-corruption baseline becomes the reference for DTB-ND1 noisy-cell ratios;
- every cell must pass typed surface route;
- every cell is evaluated with the same six frozen D2 proxy checks/margins;
- an epsilon level passes only if all four ALL8 ell cells pass;
- report the largest tested passing epsilon and the next tested failing epsilon;
- no interpolation claim.

## Required diagnostics

For every replay cell report:

- depth `|delta|` RMS and p95;
- valid-normal coverage on every set of matcher-requested rows;
- sign-invariant angular error of delivered robust `N_d` against the same robust operator on clean observable P for matched/requested rows;
- D2 support-pair count;
- all six proxy ratios/checks;
- per-family tails.

The historical stride-2 operator remains the sealed DTB-S1/DTB-S1R comparator; it is not rewritten.

## Interpretation firewall

- If clean non-inferiority passes and robust local-plane materially widens the ALL8 boundary, DTB-S1R is interpreted as substantially differential-operator-limited.
- If clean non-inferiority passes but the boundary does not widen, the current tolerance bottleneck survives the normal-operator intervention.
- If clean non-inferiority fails, the new operator is rejected in this form before capacity conclusions are drawn.
- Neither result is yet a product-safe IRIS acceptance region because the real-consumer interlock remains open.
- No DINO/foundation-capacity decision may use the old DTB-S1R bracket without considering DTB-ND1.
