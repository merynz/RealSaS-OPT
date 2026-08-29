# RealSaS — M4R ALL8 Boundary Refinement Closure

**Date:** 2026-08-29  
**Status:** `M4R_ALL8_BOUNDARY_REFINEMENT_CLOSED__PRODUCT_SAFE_NOT_CLAIMED`  
**Parent M4 result:** `9de6c8aa150cda3b333753dd58359d93bd688271e0c9e5d7aa2182b2030d90fb`  
**M4R prereg SHA-256:** `a766096623ddc07d538f2a77c2099028a264d49465666b1f7e8c4a77a5e8faaf`  
**M4R result SHA-256:** `fb155acdeffaf914071c9aa75c65cbaed5eac7aab43afe135fca633cfd53c116`

## Result

The preregistered 21-cell ALL8 refinement completed with the same frozen M4 surface route, proxy bytes, source population and decision margins.

| epsilon RMS | ell=0 | ell=4 | ell=16 | ell=64 | epsilon-level |
| ---: | :---: | :---: | :---: | :---: | :---: |
| 0.00175 | PASS | PASS | PASS | PASS | PASS |
| 0.00200 | PASS | PASS | PASS | PASS | PASS |
| 0.00225 | PASS | PASS | PASS | PASS | PASS |
| 0.00250 | FAIL | FAIL | PASS | PASS | **FAIL** |
| 0.00275 | FAIL | FAIL | FAIL | FAIL | **FAIL** |

Preregistered rule: an epsilon level passes only if all four ell cells pass.

Therefore the current frozen surface/proxy ALL8 transition is:

`0.00225 <= epsilon_critical < 0.0025`

No interpolation beyond the tested bracket is claimed.

This is still not a product-safe IRIS acceptance interval. The consumer-validity interlock remains mandatory.

## Boundary mechanism

At epsilon=.0025:

- ell=0 Arachne CE ratio: `1.051834 > 1.05` → FAIL
- ell=4 Arachne CE ratio: `1.052156 > 1.05` → FAIL
- ell=16 Arachne CE ratio: `1.044959` → PASS
- ell=64 Arachne CE ratio: `1.046970` → PASS

No Geppetto veto is responsible for this boundary.

A diagnostic replay with the current historical E0 differential-normal operator gives mean affected-view sign-invariant `theta_Nd` Q95:

- ell=0: ~66.74 deg
- ell=4: ~38.39 deg
- ell=16: ~11.06 deg
- ell=64: ~3.09 deg

Meanwhile depth `|delta| p95` stays ~0.00487–0.00490.

So ell is not an ignorable axis at the actual transition. High-frequency residual structure is materially more damaging to the current derived-normal / skin-proxy path.

Important limitation: the future V-next normal operator is planned as a robust local-plane fit, not this historical E0 stride-2 operator. The boundary must be replayed when that operator changes.

## IRIS-operational meaning

M4R is ALL8-only by design.

It answers the narrow current question:
> How much simultaneous eight-view RMS ray-depth error can the frozen surface/proxy chain tolerate under the tested spatial-correlation classes?

Asymmetric M4 columns remain separate redundancy / future Mode-E diagnostics.

## Historical scratch context

The old P-V5 R256 8x2 scratch learner's aggregate `P_p95 ~= .003706` and worst `~.005741` sit near this tolerance scale, but cannot be directly mapped to M4R RMS because:
- p95 != RMS;
- residuals are non-Gaussian;
- it was fit-only;
- real view covariance is not the synthetic generator.

Therefore the next cheap model-side fact remains a family-disjoint sacrificial IRIS residual pilot in the exact ray-depth coordinate system.

## Firewalls

- Parent M4 remains immutable.
- M4R used no Proxy27 tuning.
- DEV32 remains closed.
- No training was performed by M4R.
- No product-safe claim is authorized.
