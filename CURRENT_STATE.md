# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Canonical continuation branch:** `main`  
**Status:** `E0_PASS__N_B3_PASS_NO_EXPLICIT_N__COMPILER_IN_SYSTEM_LOOP__STRUCTURED_PREDICTED_DEPTH_BRIDGE_NEXT__DEV32_CLOSED`

This file on `main` is the single continuation authority.

All non-main branches are historical/experimental evidence only. Do not continue product/scientific work from a side branch unless `main` explicitly promotes that branch state.

## Closed scientific state

- E0 downstream information-sufficiency Proxy27: PASS; product pass not claimed; DEV32 remains closed.
- N-B3: PASS. No explicit learned normal feature is required under the frozen tested regime; direct-N is removed from the required exposed IRIS output contract.
- Required learned geometric direction is forward depth `d`; with known orthographic cameras, `P_hat = O + d_hat F` is analytic.
- B3 isotropic epsilon is NOT a product depth tolerance envelope.

## Compiler/runtime restoration — CLOSED FOR CURRENT EXECUTION

Compiler is now part of the canonical system/evaluation loop.

```text
IRIS d
 -> analytic P_hat
 -> deterministic SurfaceBuilder / MUTUAL_P003
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

## NEXT — structured ray-aligned predicted-depth tolerance bridge

Question:

> Which forward-depth error regimes can IRIS tolerate before persistence, compiler qualification or functional proof becomes unsafe?

Preregister a three-axis corruption envelope:

1. magnitude `epsilon` along each known camera ray before persistence;
2. spatial correlation length `ell` so coherent regional depth drift is tested, not only independent pixel noise;
3. view asymmetry `A` so one/few bad views can be stressed against the remaining good views.

For each frozen corruption cell:
- compute `P_hat = O + d_hat F`;
- rerun teacher-free `MUTUAL_P003`;
- evaluate the same frozen proxy-consumer bytes (no retraining per cell);
- route the same admitted `RiggingSurfaceIR` through the real Compiler path;
- record persistence error, qualification/abstention, structural validity, and available deformation/proof failures;
- inherit existing E0 practical margins where the same proxy metric applies;
- report hard per-family tails separately from aggregates.

Do not use Proxy27 or DEV32 for tuning. Do not convert the result into one scalar `P95 <= epsilon*`; freeze an admissible region over `(epsilon, ell, A)`.

## After the bridge

1. freeze compiler-qualified admissible depth-error region;
2. MapAnything/external-prior preflight against that region;
3. backbone + native high-resolution/multiview C-path bake-off;
4. train depth-only IRIS;
5. compare actual held-out structured residuals to the frozen region;
6. run final `d_hat -> P_hat -> persistence -> Compiler -> puppet` qualification.

Deferred corpus product-hygiene work remains prospective and must not rewrite sealed E0/N-B3 populations.
