# RealSaS N-B3 — Canonical Qualification Report V1

**Date:** 2026-08-28  
**Status:** `N_B3_NO_EXPLICIT_NORMAL_FEATURE_NEEDED_OVER_FROZEN_EPSILON_LADDER`  
**Scope:** FIT-only information-value qualification; not product IRIS qualification.

## Authorities

- Operational freeze SHA-256: `d1c8d95417074f7bca879b6d46a3ee2696141dfb39ec31cad43f58999e9efb08`
- Operational spec SHA-256: `7b633513e062586f5cdb1edc91bf3d2c49d4e7979acb2e3f1fd3a64010b2c9f9`
- Runner SHA-256: `ff92220ce7157968b1fbaf76670d2cefd46ed7f2ef776966841134083ab16701`
- N3v method-selection SHA-256: `9b7b52ba1bca4b526f093064d0d2ba0b68c2620eb4fbaf2d170bb3c11171e63e`
- Prequalification seal SHA-256: `0b541da42aa4970b90d45fb0988f88777744015318f937882b14d2da0dd826a0`
- Qualification64 ordered-ID SHA-256: `dffee4dd926eef5f08f80c1cc956d2284bb6d8dfb418f2bf88a99cefdbe93644`
- Result SHA-256: `5acbcab3cb19b3073a704e8d2ee1b391f1b390720a708ef256f887ba8b929502`
- Final-seal file SHA-256: `b0390c9e8d00c9fa6b70df9c99561697efcb68e38229e31eecb3e0b96764789e`

## Firewall audit

- B3 train / selection / qualification: `310 / 59 / 64`, pairwise disjoint.
- All 64 qualification assets came from the outcome-blind final-64 partition of the frozen E0 train order.
- N3v method selection occurred on selection59 before qualification64 opened.
- All `5 arms × 4 eps × 2 consumers = 40` checkpoints were trained, selected and hash-sealed before qualification open.
- `adaptation_after_open_forbidden = true`.
- Proxy27 used: `false`.
- DEV32 opened: `false`.
- Official 437 pack rebuilds: `0`.
- E0 Proxy27 result was not reinterpreted.

## Selected deterministic normal challenger

`N3v = k8 + VISIBILITY_THEN_MASS`.

Selection59 geometric fidelity:

| epsilon | oriented median | oriented p90 | sign-invariant median |
|---:|---:|---:|---:|
| 0.000 | 14.39° | 58.60° | 13.56° |
| 0.010 | 18.43° | 63.41° | 17.53° |

This is deliberately not interpreted as an exact-normal reconstruction. The experiment asks whether the explicit normal information adds downstream value beyond X36.

## Qualification64 — N3v vs exact-N upper bound N2

Inherited E0 gates: primary ratios `<=1.05`, Geppetto family-p95 ratio `<=1.10`, PCK deltas `>=-0.10`.

| epsilon | Arachne CE | Arachne influence | Geppetto joint mean | Geppetto family-p95 | PCK@.05 Δ | PCK@.08 Δ | decision |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.000 | 1.001753 | 1.000897 | 1.025641 | 0.949169 | +0.006100 | +0.007934 | PASS |
| 0.001 | 1.002058 | 1.000979 | 1.004975 | 0.980320 | +0.004813 | +0.019655 | PASS |
| 0.003 | 1.001667 | 1.000792 | 1.018630 | 0.933349 | +0.025549 | +0.033525 | PASS |
| 0.010 | 1.000719 | 1.001266 | 0.995356 | 0.995266 | +0.020429 | +0.029856 | PASS |

N3v passes all six checks at all four epsilon values. The narrowest N3v primary gate is Geppetto joint mean at epsilon 0: `1.025641` against `1.05`.

## Stronger finding — N0 also passes

The zero-explicit-normal arm `N0 = X36 + zeroed N slot` is also non-inferior to N2 at all four epsilon values. Its narrowest primary gate is Geppetto joint mean at epsilon `0.003`: `1.034508` against `1.05`.

Therefore the primary B3 interpretation follows the preregistered first branch:

> Under the frozen B3 population, epsilon ladder, matched-capacity consumers and inherited non-inferiority margins, no explicit normal feature is required for downstream information sufficiency.

This is stronger than merely showing that deterministic N3v can substitute for exact N.

## Secondary tail audit

Population-level and preregistered tail gates pass, but individual Geppetto families can still move materially between N3v and N2. This does not invalidate the preregistered decision, but it forbids the stronger claim that normals are useless for every morphology/family.

The result therefore supports removing an explicit learned/direct N channel from the canonical IRIS output contract. It does **not** prove:

- exact or deterministic normals are useless for every individual family;
- normal-derived compiler geometry operations are unnecessary;
- an auxiliary normal loss during training can never be useful;
- predicted depth/P is already accurate enough;
- product Geppetto/Arachne are qualified.

## Architecture consequence

Canonical IRIS geometry can now proceed with:

```text
known cameras + raster/support
    -> learned forward depth d
    -> analytic P = O + d F
    -> deterministic SurfaceBuilder / MUTUAL_P003
    -> compiler consumers
```

No explicit learned `N` field is justified as a required exposed geometric output by N-B3. If a normal is useful internally, it should be derived deterministically from the observable substrate unless a later product-specific falsification reopens the question.

## Next gate

`RAY_ALIGNED_PREDICTED_P_DEPTH_TOLERANCE_BRIDGE`

The next experiment must not reuse B3 isotropic 3D epsilon as a proxy for product inference error. It must perturb forward depth along the known camera ray **before** persistence construction, compile `P_hat = O + d_hat F`, rerun teacher-free `MUTUAL_P003`, and evaluate frozen-consumer robustness. The purpose is to derive a maximum admissible depth-error envelope before training the final depth-only IRIS.
