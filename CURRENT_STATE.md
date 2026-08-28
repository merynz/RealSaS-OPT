# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS__N_B3_PASS_NO_EXPLICIT_N__PREDICTED_P_DEPTH_BRIDGE_NEXT__DEV32_CLOSED`

This file is the single continuation authority.

## Closed scientific results

### E0 downstream information sufficiency — PASS

The frozen Proxy27 qualification established that exact observable A×8 common-frame `P`, observation-derived support/provenance and teacher-free `MUTUAL_P003` persistence preserve enough rigging-relevant information under the matched information-isolation consumers and frozen non-inferiority margins.

- Proxy27 count: 27
- qualification result SHA-256: `86153066f487bc16fdfd4e817494fa3bd052def076ef97cd998c013455ee74e5`
- seal SHA-256: `0bbd00843436e78c4cc38cb50765295ba3ba7844c57b7ca7cb23fe3f7c57fe78`
- all frozen checks: `14/14 PASS`
- DEV32: closed
- product pass: not claimed

Reusable E0 substrate remains immutable:

- `prep_v1_2`: 437 packs
- geometry authority SHA-256: `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`
- scientific builder SHA-256: `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`
- D2 persistence: `MUTUAL_P003`, threshold `0.003`, teacher identity forbidden

The closest E0 primary gate remains Arachne CE D2/D1 = `1.039926` against `1.05`; this is an important sensitivity target for the predicted-P bridge.

### Pre-Proxy diagnostics — CLOSED

X36 carries substantial implicit orientation information: oriented normal median/p90/p95 `24.95° / 64.27° / 77.91°`. D0@512→D0@2048 greatly improved geometric coverage while frozen fixed-capacity downstream probes moved negligibly. These remain interpretation-only results.

### N-B3 explicit-normal marginal value — PASS

Canonical report:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/N_B3_CANONICAL_REPORT_V1.md`

Authorities:

- operational freeze SHA-256: `d1c8d95417074f7bca879b6d46a3ee2696141dfb39ec31cad43f58999e9efb08`
- spec SHA-256: `7b633513e062586f5cdb1edc91bf3d2c49d4e7979acb2e3f1fd3a64010b2c9f9`
- selected N3v: `k=8`, `VISIBILITY_THEN_MASS`
- method-selection SHA-256: `9b7b52ba1bca4b526f093064d0d2ba0b68c2620eb4fbaf2d170bb3c11171e63e`
- prequalification seal SHA-256: `0b541da42aa4970b90d45fb0988f88777744015318f937882b14d2da0dd826a0`
- qualification64 SHA-256: `dffee4dd926eef5f08f80c1cc956d2284bb6d8dfb418f2bf88a99cefdbe93644`
- result SHA-256: `5acbcab3cb19b3073a704e8d2ee1b391f1b390720a708ef256f887ba8b929502`
- status: `N_B3_NO_EXPLICIT_NORMAL_FEATURE_NEEDED_OVER_FROZEN_EPSILON_LADDER`

Firewalls: train/selection/qualification `310/59/64` disjoint; 40/40 checkpoints sealed before qualification open; Proxy27 unused; DEV32 closed; official pack rebuilds 0; no adaptation after qualification open.

Key result:

- N3v is non-inferior to exact-N N2 at all four epsilon values `{0,.001,.003,.010}`.
- More strongly, N0 (no explicit normal feature) is also non-inferior to N2 at all four epsilon values.
- N3v narrowest primary ratio: Geppetto joint mean `1.025641` at epsilon 0 vs max `1.05`.
- N0 narrowest primary ratio: Geppetto joint mean `1.034508` at epsilon `.003` vs max `1.05`.

Frozen interpretation:

> No explicit normal feature is required for downstream information sufficiency under the tested B3 regime. A learned direct-N head is therefore removed from the required canonical IRIS output contract.

Caveat: individual families can still benefit from N; B3 is a population-level information-sufficiency result under fixed proxies. Deterministic normals may still be derived internally for compiler geometry operations, and an auxiliary training-only normal objective is not forbidden.

## Current IRIS geometry contract direction

The required exposed canonical geometric field is now narrowed to forward depth:

```text
8x raster + known orthographic cameras
  -> external pretrained spatial prior B
  -> native high-resolution/multiview adapter C
  -> learned forward depth d
  -> analytic P_hat = O + d_hat * F
  -> deterministic SurfaceBuilder / MUTUAL_P003
  -> RiggingSurfaceIR / compiler
```

Not required as exposed learned geometric outputs:

- `N`: closed by N-B3
- `P`: analytic from known ray + learned depth
- visibility/support: deterministic raster authority
- `Z` heads: no downstream consumer justification
- public uncertainty/logsigma: not justified unless a later adaptive-consumer probe gives it a concrete use

## NEXT — ray-aligned predicted-P/depth tolerance bridge

Question:

> How much forward-depth error can product IRIS tolerate before `P_hat -> MUTUAL_P003 -> downstream` violates the already-frozen practical margins?

Required experiment shape:

1. Use known cameras and exact observable per-view points as the clean depth authority.
2. Inject deterministic forward-depth error **along the camera ray before persistence matching**.
3. Compile `P_hat = O + d_hat F`.
4. Re-run teacher-free `MUTUAL_P003`; no teacher identity or exact correspondence may rescue persistence.
5. Freeze clean-trained downstream consumers, then evaluate the same consumer bytes across a preregistered depth-error ladder. Do not retrain per error rung; this is a robustness bridge, unlike B3.
6. Inherit the E0 practical non-inferiority margins; derive the largest contiguous passing depth-error envelope.
7. Report persistence degradation and per-family hard tails as diagnostics.
8. Keep Proxy27 and DEV32 closed unless a dedicated prereg explicitly authorizes a later external confirmation.

The B3 isotropic 3D epsilon ladder must **not** be interpreted as the product depth envelope.

## After the depth bridge

1. Freeze the admissible depth-error envelope.
2. Backbone bake-off: pinned Apache-compatible external prior(s) + native C-path capacity ladder.
3. Train depth-only IRIS under known-camera / legal-depth constraints.
4. Measure actual held-out `d_hat` error against the frozen envelope.
5. Run final `P_hat -> MUTUAL_P003 -> downstream` qualification.
6. Freeze the IRIS product contract only if the predicted-depth bridge passes.

## Deferred corpus hygiene

Product character/presentation hygiene remains a prospective corpus task before broad product-depth training. It must not rewrite sealed E0 or N-B3 populations/results.

## Firewall state

`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS = TRUE`

`N_B3_EXPLICIT_NORMAL_REQUIRED = FALSE`

`N_B3_RESULT = PASS__NO_EXPLICIT_NORMAL_FEATURE_NEEDED`

`PREDICTED_P_DEPTH_BRIDGE = NEXT__REQUIRED`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`
