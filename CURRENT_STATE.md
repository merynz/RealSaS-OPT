# RealSaS-OPT — Current State

**Date:** 2026-08-29  
**Canonical continuation branch:** `main`  
**Status:** `E0_PASS__N_B3_PASS_NO_EXPLICIT_N__COMPILER_IN_SYSTEM_LOOP__DTB_S1_CLOSED__DTB_S1R_CLOSED__DTB_ND1_CLOSED__FIT_PROXY32_RESIDUAL_AUDIT_NEXT__DEV32_CLOSED`

This file on `main` is the single continuation authority.

All non-main branches are historical/experimental evidence only. Do not continue product/scientific work from a side branch unless `main` explicitly promotes that branch state.

## Experiment naming authority

To avoid collision with the older IRIS qualification line:

- `LEGACY-IRIS-M4` = historical IRIS qualification work from the older line.
- `DTB-S1` = the current structured depth bridge whose sealed files retain `M4_*` names.
- `DTB-S1R` = the ALL8 boundary refinement whose sealed files retain `M4R_*` names.
- `DTB-ND1` = the robust local-plane normal intervention.

Sealed historical filenames are not renamed.

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

### Route-semantics correction — CLOSED BEFORE DTB-S1 OUTCOME OPEN

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
- scientific DTB-S1 qualification uses post-cycle `D2_MUTUAL_P003`;
- BASE is retained only as a separate mechanism diagnostic.

Additive cached/vectorized runtime parity:
- zero-corruption BASE `1909 == 1909`, bit-exact;
- zero-corruption D2 `1884 == 1884`, `D2_X` bit-exact;
- nonzero vectorized forward matcher: 56/56 scalar row comparisons exact;
- corruption CPU tests remain 5/5 PASS.

## DTB-S1 structured surface/substrate grid — CLOSED (sealed files retain M4 names)

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

Under the frozen historical D2 proxy consumer:
- `epsilon = 0.0015`: ALL8 PASS for `ell = 0,4,16,64`;
- `epsilon = 0.003`: ALL8 FAIL for `ell = 0,4,16,64`.

Therefore the coarse transition is:

`0.0015 <= epsilon_critical < 0.003`

where epsilon is RMS ray-aligned depth displacement on every affected view.

This is NOT a final product-safe IRIS tolerance. Product interpretation still requires the consumer-validity interlock.

### ell interpretation

Binary verdict changes across ell in only one tested `(epsilon, asymmetry)` group:
`epsilon=.006, TWOOPP`.

However ell is not mechanically dead. A post-hoc diagnostic using the historical E0 differential-normal operator shows strong `theta_Nd` dependence at that slice.

### Proxy failure localization

All 25 proxy-FAIL cells still pass the typed surface route.

Frozen check failure counts:
- Arachne CE +5%: 25/25 FAIL cells;
- Arachne influence-displacement +5%: 20/25;
- Geppetto family-P95 +10%: 3/25.

This is evidence about the frozen historical D2 proxy chain, not future production Geppetto/Arachne.

## DTB-S1R ALL8 boundary refinement — CLOSED (sealed files retain M4R names)

Canonical closure: `canonical/M4R_ALL8_BOUNDARY_REFINEMENT_CLOSURE_20260829.md`.

The preregistered ALL8 refinement closed the historical-normal/frozen-proxy bracket to:

`0.00225 <= epsilon_critical < 0.0025`

Detailed verdict:
- 0.00175: 4/4 ell PASS
- 0.00200: 4/4 ell PASS
- 0.00225: 4/4 ell PASS
- 0.00250: ell=0/4 FAIL, ell=16/64 PASS -> epsilon-level FAIL
- 0.00275: 4/4 ell FAIL

At that boundary the firing check was Arachne CE under the frozen D2 proxy. Parent DTB-S1 is immutable. DTB-S1R is not a product-safe tolerance claim.

## DTB-ND1 robust local-plane intervention — CLOSED

Canonical closure: `canonical/DTB_ND1_ROBUST_LOCAL_PLANE_CLOSURE_20260829.md`.

DTB-ND1 changed only the differential normal estimator used by persistence matching. `P` is not fitted, moved, or smoothed; the robust plane emits `N_d` only and changes downstream evidence through normal-gated correspondence/admission.

Clean non-inferiority was preregistered before noisy outcomes and passed all six frozen proxy checks.

Closed ALL8 result:
- 0.00175: 4/4 ell PASS
- 0.00200: 4/4 ell PASS
- 0.00225: 4/4 ell PASS
- 0.00250: 4/4 ell PASS
- 0.00275: ell=0 PASS, ell=4/16/64 FAIL -> epsilon-level FAIL

Current frozen historical-D2-proxy bracket:

`0.00250 <= epsilon_critical < 0.00275`

At the new boundary the only firing frozen check is Arachne CE +5%; Geppetto checks and Arachne influence-displacement remain PASS.

Mechanism correction:
- robust `N_d` materially improves differential-normal stability at the rescued `.0025` high-frequency cells;
- however `theta_Nd` alone does not determine verdict (`.00275/ell=0` has worse theta than `.00275/ell=16` but passes while ell=16 fails);
- the supported causal statement is that robust normal estimation changes normal-gated matching/support topology and thereby improves tolerance.

Binding Arachne checkpoint:
`72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745`

DTB-ND1 is not a product-safe tolerance claim.

### Stop rule after DTB-ND1

No further deterministic operator tuning is allowed before the model-side residual/generalization fact is measured.

Forbidden until the residual audit closes:
- changing plane-fit window;
- changing MAD threshold;
- changing minimum neighbors;
- changing hull dilation;
- changing persistence thresholds;
- opening another cheap geometry tolerance intervention.

NEXT is the FIT_PROXY32 coordinate/target equivalence audit and held-out ray-depth residual measurement.

## Residual-scale pilot before expensive foundation selection

A family-disjoint residual audit is required before interpreting the DINO capacity ladder.

Historical context only:
- old P-V5 R256 scratch 8x2 fit-only learner ended at aggregate `P_p95 ~= 0.003706`;
- worst cell `P_p95 ~= 0.005741`.

Matched-statistic context for the tested Gaussian-like ell=0 bridge:
- DTB-ND1 lower bracket endpoint epsilon=.00250 corresponds to depth `abs-p95 ~= .00490`;
- upper tested failing level epsilon=.00275 corresponds to depth `abs-p95 ~= .00539`.

The historical fit number is therefore below this matched p95 scale, but it is not held-out evidence and coordinate equivalence must still be audited.

The next gate must first audit FIT_PROXY32 coordinate/target equivalence, then measure held-out ray-depth residual distributions and cross-view structure without changing frozen DTB-S1/DTB-S1R/DTB-ND1 consumer criteria.

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

## After FIT_PROXY32 residual audit + consumer-validity interlock

1. freeze the consumer-qualified admissible depth-error region;
2. MapAnything/external-prior preflight against that region;
3. backbone + native high-resolution/multiview C-path bake-off;
4. train depth-only IRIS;
5. compare actual held-out structured residuals to the frozen region;
6. run final `d_hat -> P_hat -> persistence -> Compiler -> puppet` qualification.

Prospective ObservationContract / foundation-prior / external-camera work remains sealed under `prospective/` and is not current scientific authority.
Deferred corpus product-hygiene work remains prospective and must not rewrite sealed E0/N-B3 populations.
