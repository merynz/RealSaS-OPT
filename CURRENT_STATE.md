# RealSaS-OPT — Current State

**Date:** 2026-08-29  
**Canonical continuation branch:** `main`  
**Status:** `E0_PASS__N_B3_PASS_NO_EXPLICIT_N__COMPILER_IN_SYSTEM_LOOP__M4_SURFACE_GRID_CLOSED__ALL8_REFINEMENT_NEXT__DEV32_CLOSED`

This file on `main` is the single continuation authority.

All non-main branches are historical/experimental evidence only. Do not continue product/scientific work from a side branch unless `main` explicitly promotes that branch state.

## Closed scientific state

- E0 downstream information-sufficiency Proxy27: PASS; product pass not claimed; DEV32 remains closed.
- N-B3: PASS. No explicit learned normal feature is required under the frozen tested regime; direct-N is removed from the required exposed IRIS output contract.
- Required learned geometric direction is forward depth `d`; with known orthographic cameras, `P_hat = O + d_hat F` is analytic.
- B3 isotropic epsilon is NOT a product depth tolerance envelope.

## Compiler/runtime restoration — CLOSED FOR CURRENT EXECUTION

Compiler is part of the canonical system/evaluation loop.

```text
IRIS d
 -> analytic P_hat
 -> deterministic SurfaceBuilder / D2_MUTUAL_P003
 -> RiggingSurfaceIR S
 -> Geppetto SkeletonProposalIR G*
 -> CanonicalGraphOptimizationRequest/Result
 -> Compiler QualifiedSkeletonIR G
 -> Arachne SkinProposalIR W*
 -> Compiler QualifiedSkinIR W
 -> CanonicalPuppetGraph Y
 -> proof / bounded repair / mandatory re-proof
 -> runtime projection
```

Important invariants:
- proposal IDs never become product-canonical IDs;
- Compiler owns root/hierarchy qualification and canonical ID minting;
- stale surface/skeleton bindings fail closed;
- proof binds exact `product_state_hash`;
- runtime export requires a passing proof for that exact state;
- no duplicate hierarchy/topology layer was created.

Current GitHub-contained historical execution closure:
- raw ZIP SHA-256 `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`
- 9 byte-exact v0.5 leaf modules + 4 namespace rebinds
- old front-brain/truth imports: 0

Heavy historical authorities remain SHA-bound in Drive:
- v0.5 full source `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`
- R5_3 `6224661cb4323f78a9b808af10f68dd584431a422e28d26a69e87816a3b0ef80`
- v97_43 `09a94871f938b069ba5c8219f203355e724f2f58afa6e10dc5c6148d98b43efb`
- C++17 runtime source `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`

Restoration regressions: current typed/routing 10/10 PASS; selected historical 46/46 PASS; causal diagnosis/repair 9/9 PASS; native runtime CTest 1/1 PASS; package-to-native render PASS.

## Structured predicted-depth bridge apparatus — M0–M3 CLOSED

Canonical closure: `canonical/STRUCTURED_DEPTH_BRIDGE_APPARATUS_CLOSURE_20260829.md`.

Closed apparatus facts:
- ray-aligned corruption CPU tests: 5/5 PASS;
- one real already-open E0 calibration family epsilon=0 replay is identity-equivalent to sealed E0-B BASE persistence;
- source rows, support, matched rows and source grid are exact at epsilon zero;
- typed Compiler surface accepts the replay with 512 surface nodes;
- three sealed non-binding smoke cells traverse independent/coherent and symmetric/asymmetric corruption through the pre-cycle teacher-free BASE matcher and typed Compiler surface;
- Proxy27 was not used for apparatus tuning;
- DEV32 remains closed.

The M3 smoke values are apparatus evidence only and are forbidden from defining the scientific tolerance grid.

### Route-semantics correction — CLOSED BEFORE M4 OUTCOME OPEN

Canonical correction: `experiments/g0_g1_single_pose_geometry/depth_tolerance_bridge_20260829/APPARATUS_ROUTE_SEMANTICS_CORRECTION_V1.md`.

The M2/M3 apparatus used `MUTUAL_P003` as an overloaded label for direct `derived_match_row` persistence. Exact downstream authority establishes two distinct stages:

```text
BASE_DERIVED_MATCHER
 -> reciprocal reverse match
 -> admit iff cycle_P <= 0.003
 -> D2_MUTUAL_P003
```

Consequences:
- sealed M2/M3 result bytes are preserved;
- their `1909 / 1921 / 1910 / 1896` support-pair values are pre-cycle BASE apparatus evidence;
- scientific M4 qualification uses post-cycle `D2_MUTUAL_P003`;
- BASE is retained only as a separate mechanism diagnostic.

Additive cached/vectorized M4 runtime parity:
- zero-corruption BASE `1909 == 1909`, bit-exact;
- zero-corruption D2 `1884 == 1884`, `D2_X` bit-exact;
- nonzero vectorized forward matcher: 56/56 scalar row comparisons exact;
- corruption CPU tests remain 5/5 PASS.

## M4 structured surface/substrate grid — CLOSED

Canonical closure: `canonical/M4_SURFACE_GRID_CLOSURE_20260829.md`.

Canonical full result SHA-256:
`9de6c8aa150cda3b333753dd58359d93bd688271e0c9e5d7aa2182b2030d90fb`

Closed facts:
- frozen grid: 65/65 cells complete;
- typed surface route: 65/65 PASS;
- frozen proxy verdict: 40 PASS / 25 FAIL;
- zero baseline BASE support pairs: 14,440;
- zero baseline post-cycle D2 support pairs: 14,094;
- Proxy27 was not used for grid tuning;
- DEV32 remains closed;
- training steps: 0.

### Primary IRIS-operational interpretation

For IRIS, the primary corruption column is `ALL8` because IRIS predicts all eight views.

`ONE`, `TWOOPP`, and `FOURALT` are retained as redundancy/external-input stress diagnostics, not as the primary IRIS acceptance columns.

Under the current frozen surface/proxy consumer:
- `epsilon = 0.0015`: ALL8 PASS for `ell = 0,4,16,64`;
- `epsilon = 0.003`: ALL8 FAIL for `ell = 0,4,16,64`.

Therefore the current coarse transition is:

`0.0015 <= epsilon_critical < 0.003`

where epsilon is RMS ray-aligned depth displacement on every affected view.

This is NOT a final product-safe IRIS tolerance. Product interpretation still requires the consumer-validity interlock.

### ell interpretation

Binary verdict changes across ell in only one tested `(epsilon, asymmetry)` group:
`epsilon=.006, TWOOPP`.

However ell is not mechanically dead. A post-hoc diagnostic using the current sealed E0 differential-normal operator shows strong `theta_Nd` dependence at that slice:
- ell=0: affected-view mean Q95 ~85.8 deg;
- ell=4: ~71.3 deg;
- ell=16: ~25.9 deg;
- ell=64: ~7.2 deg.

The ell=0 TWOOPP failure is caused by frozen Arachne CE crossing `1.054566 > 1.05`; ell>=4 passes that margin.

Important limitation: this diagnostic uses the historical E0 stride-2 local tangent/cross-product normal operator, not the prospective V-next robust local-plane operator.

### Proxy failure localization

All 25 proxy-FAIL cells still pass the typed surface route.

Frozen check failure counts:
- Arachne CE +5%: 25/25 FAIL cells;
- Arachne influence-displacement +5%: 20/25;
- Geppetto family-P95 +10%: 3/25.

This is evidence about the frozen historical D2 proxy chain, not future production Geppetto/Arachne.

## NEXT — M4R ALL8 boundary refinement

Parent M4 is immutable.

Preregistered M4R asks only where inside `[0.0015, 0.003)` the current ALL8 transition lies.

Frozen follow-up:
- asymmetry: ALL8 only;
- epsilon RMS: `0.00175, 0.002, 0.00225, 0.0025, 0.00275`;
- ell: `0,4,16,64`;
- one zero baseline;
- 21 cells total;
- same frozen surface route and six proxy checks;
- an epsilon level passes only if all four ell cells pass.

No M4R outcome may alter M4.

## Residual-scale pilot before expensive foundation selection

A family-disjoint sacrificial IRIS residual pilot is required before interpreting the DINO capacity ladder.

Historical context only:
- old P-V5 R256 scratch 8x2 fit-only learner ended at aggregate `P_p95 ~= 0.003706`;
- worst cell `P_p95 ~= 0.005741`.

That historical fit scale lies inside/near the coarse M4 ALL8 bracket after accounting for RMS-vs-p95, but it is not held-out evidence and is not qualification.

The sacrificial pilot must measure actual held-out ray-depth residual distributions and cross-view structure without changing the frozen M4/M4R consumer criteria.

## Product-level interpretation interlock

The surface/substrate bridge is not automatically the final product-safe IRIS acceptance region.

Before using it to select the final foundation/IRIS model, close the sealed consumer-validity interlock with minimal real consumers on the actual RealSaS substrate:

```text
ClosedRiggingVolume V0
 -> InteriorRiggingSubstrate V0
 -> Geppetto G0
 -> Compiler QualifiedSkeleton
 -> Arachne A0
 -> Compiler QualifiedSkin
 -> deformation / proof replay
```

Product-level tolerance is consumer-profile specific. Retraining/replacing Geppetto or Arachne creates a new consumer profile and requires replay before old model rankings transfer.

## After M4R + residual-scale pilot + consumer-validity interlock

1. freeze the consumer-qualified admissible depth-error region;
2. MapAnything/external-prior preflight against that region;
3. backbone + native high-resolution/multiview C-path bake-off;
4. train depth-only IRIS;
5. compare actual held-out structured residuals to the frozen region;
6. run final `d_hat -> P_hat -> persistence -> Compiler -> puppet` qualification.

Prospective ObservationContract / foundation-prior / external-camera work remains sealed under `prospective/` and is not current scientific authority.
Deferred corpus product-hygiene work remains prospective and must not rewrite sealed E0/N-B3 populations.
