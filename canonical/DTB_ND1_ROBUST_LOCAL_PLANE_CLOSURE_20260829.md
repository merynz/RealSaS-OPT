# RealSaS — DTB-ND1 Robust Local-Plane Closure

**Date:** 2026-08-29  
**Status:** `DTB_ND1_CLOSED__NO_MORE_OPERATOR_TUNING_BEFORE_RESIDUAL_PILOT`

## Identity

Naming authority:
- `LEGACY-IRIS-M4` = old historical IRIS qualification line.
- `DTB-S1` = current 65-cell structured depth bridge whose sealed files retain `M4_*` names.
- `DTB-S1R` = ALL8 boundary refinement whose sealed files retain `M4R_*` names.
- `DTB-ND1` = this robust local-plane normal intervention.

Sealed historical filenames are not renamed.

## Intervention

DTB-ND1 changed only the differential normal estimator used by persistence matching.

Frozen operator:
- 7x7 stride-2 support over +/-6 native pixels;
- Gaussian spatial weights;
- MAD-based 3D neighbor rejection;
- minimum 6 retained neighbors;
- weighted local covariance;
- smallest-eigenvector plane normal;
- no historical-normal fallback.

`P` is **not fitted, moved, or smoothed** by this operator. The fitted plane emits `N_d` only. Its influence on the proxy path is through normal-gated correspondence/admission, which changes D2 support/raster evidence and therefore X36.

Depth corruption, source rows, matcher thresholds, reciprocal cycle threshold, consumer checkpoints, proxy metrics and margins are unchanged.

## Clean non-inferiority

Before noisy outcomes were opened, robust-zero was required to pass the six frozen historical-zero proxy margins.

Result: **6/6 PASS**.

- historical-zero D2 support pairs: 14,094
- robust-zero D2 support pairs: 14,166
- robust normal valid coverage: ~99.997%

Thus tolerance improvement is not obtained by accepting a degraded clean baseline.

## Boundary result

| epsilon RMS | ell=0 | ell=4 | ell=16 | ell=64 | epsilon-level |
| ---: | :---: | :---: | :---: | :---: | :---: |
| .00175 | PASS | PASS | PASS | PASS | PASS |
| .00200 | PASS | PASS | PASS | PASS | PASS |
| .00225 | PASS | PASS | PASS | PASS | PASS |
| .00250 | PASS | PASS | PASS | PASS | **PASS** |
| .00275 | PASS | FAIL | FAIL | FAIL | **FAIL** |

Same prereg rule as DTB-S1R: an epsilon level passes only if all four ell cells pass.

Therefore under the same frozen historical D2 proxy consumer:

`0.00250 <= epsilon_critical < 0.00275`

This advances the DTB-S1R bracket `[.00225,.00250)` but does not authorize a product-safe IRIS tolerance claim.

Matched p95 scale for the tested Gaussian-like ell=0 perturbation is roughly:
- epsilon=.00250 -> depth abs-p95 ~= .00490
- epsilon=.00275 -> depth abs-p95 ~= .00539

## Mechanism — corrected causal statement

The earlier hypothesis that verdict was controlled directly by `theta_Nd` is too strong.

At epsilon=.00250:
- ell=0 historical E0 operator: theta_Nd Q95 ~= 66.7 deg, FAIL
- ell=0 robust operator: mean view/asset theta_Nd Q95 ~= 30.3 deg, PASS
- ell=4 historical ~= 38.4 deg, FAIL
- ell=4 robust ~= 29.8 deg, PASS

So robust fitting materially improves differential-normal stability at the rescued cells.

However theta accuracy alone does **not** determine verdict:
- epsilon=.00275, ell=0: robust theta_Nd Q95 ~= 37.6 deg -> PASS
- epsilon=.00275, ell=16: robust theta_Nd Q95 ~= 11.8 deg -> FAIL
- epsilon=.00275, ell=64: robust theta_Nd Q95 ~= 3.4 deg -> FAIL

Therefore the supported causal statement is:

`robust N_d changes normal-gated correspondence/admission -> changes D2 support/evidence topology -> increases tolerance`.

It is not justified to claim:
`lower theta_Nd alone -> PASS`.

There is no literal P-surface smoothing in DTB-ND1; P is unchanged.

## New binding constraint

At the new epsilon=.00275 boundary:
- ell=0 passes all six checks;
- ell=4,16,64 fail **only** `Arachne CE +5%`;
- Geppetto checks remain PASS;
- Arachne influence-displacement remains PASS.

The binding learned proxy is therefore still the frozen historical D2 Arachne isolation proxy:

`SHA-256 72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745`

This provenance must accompany every use of the DTB-ND1 bracket.

## Stop rule

DTB-ND1 closes the deterministic-operator intervention phase.

**No additional window-size, MAD-threshold, neighbor-count, hull, persistence-threshold, or other cheap tolerance tuning is allowed before the residual-scale/generalization pilot.**

The next execution gate is:
1. FIT_PROXY32 coordinate/target equivalence audit;
2. recompute held-out residuals in exact ray-depth units;
3. disaggregate by family/source/view and measure covariance/error shape;
4. only then decide whether the DINO capacity ladder is warranted.

Any future operator intervention is a new versioned experiment opened only after that model-side fact is known.
