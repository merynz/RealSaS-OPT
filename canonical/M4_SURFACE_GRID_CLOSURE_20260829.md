# RealSaS — M4 Structured Depth Surface Grid Closure

**Date:** 2026-08-29  
**Status:** `M4_SURFACE_GRID_CLOSED__PRODUCT_SAFE_NOT_CLAIMED`  
**Canonical full-result SHA-256:** `9de6c8aa150cda3b333753dd58359d93bd688271e0c9e5d7aa2182b2030d90fb`  
**Canonical summary SHA-256:** `9f2c0e8ce2a73ecce1de8f4c9eff46e0dd5d11b72da17c54621e25c39f9c7156`  
**Mechanism diagnostic SHA-256:** `2f6ae6169a4b6f6418b84fe26cce35a3cd60396836bf65460d7a0075011b7ad8`

## Closed result

The frozen 65-cell structured depth grid completed under the sealed M4 runner.

- cells: **65 / 65**
- typed surface route: **65 / 65 PASS**
- frozen proxy verdict: **40 PASS / 25 FAIL**
- baseline BASE support pairs: **14,440**
- baseline post-cycle D2 support pairs: **14,094**
- no Proxy27 tuning
- DEV32 closed
- training steps: 0

The resumable execution copy and the direct frozen-runner canonical output have identical 65-cell result payloads.

## Primary scientific interpretation

For IRIS itself, the operational corruption column is **ALL8**.

IRIS predicts all eight views, so `ONE`, `TWOOPP`, and `FOURALT` are not the primary IRIS acceptance columns. They are retained as redundancy and external-input stress diagnostics and are especially relevant to future ObservationContract Mode-E handling.

Under the current frozen surface/proxy consumer:

- `epsilon = 0.0015`: ALL8 passes for `ell = 0,4,16,64`.
- `epsilon = 0.003`: ALL8 fails for `ell = 0,4,16,64`.

Therefore the current coarse ALL8 transition lies in:

`0.0015 <= epsilon_critical < 0.003`

where epsilon is per-affected-view RMS ray-aligned depth displacement.

This bracket is **not** a final product-safe IRIS acceptance region. The consumer-validity interlock remains mandatory.

## Why asymmetric columns still matter

The asymmetric columns demonstrate real eight-view redundancy:

- ONE remains proxy-PASS through `epsilon = 0.012` over all tested ell values.
- TWOOPP and FOURALT fail earlier.
- At `epsilon = 0.006`, TWOOPP changes from FAIL at `ell=0` to PASS for `ell>=4`.

These columns answer a different question from ALL8: how much local/view-specific corruption can the current multiview persistence/consumer path absorb when other views remain good?

Do not use ONE's high tolerance as an IRIS depth budget.

## ell is low-sensitivity at verdict level, not dead mechanistically

Only one `(epsilon, asymmetry)` group changes binary verdict across ell:
`epsilon=0.006, TWOOPP`.

However a post-hoc, non-authoritative mechanism diagnostic using the **current sealed E0 normal operator** shows strong ell dependence at that exact slice:

- ell=0: mean affected-view `theta_Nd p95 ~= 85.8 deg`
- ell=4: `~71.3 deg`
- ell=16: `~25.9 deg`
- ell=64: `~7.2 deg`

Depth `|delta| p95` stays approximately constant near 0.0117.

The only frozen check that fails at `epsilon=.006, ell=0, TWOOPP` is Arachne CE:
`1.054566 > 1.05`.

This supports the mechanism hypothesis:

`high-frequency depth noise -> degraded derived differential geometry -> skin-proxy sensitivity`.

Important limitation: M4 uses the historical E0 stride-2 local tangent/cross-product normal operator, **not** the prospective V-next robust local-plane `N_d`. This diagnostic must not be generalized to the future normal operator without replay.

## Failure localization

All 25 proxy-FAIL cells still pass the typed surface route.

Frozen check failure counts:

- Arachne CE +5% margin: **25 / 25 FAIL cells**
- Arachne influence-displacement +5% margin: **20 / 25**
- Geppetto family-P95 +10% margin: **3 / 25**

Thus the present proxy chain is more sensitive at the skin-proxy stage than at the skeleton-proxy stage.

This is a statement about the frozen historical D2 proxies, not about future production Geppetto/Arachne.

## Historical learner scale context — diagnostic only

The historical P-V5 R256 scratch 8x2 joint-fit learner ended at:

- aggregate `P_p95 ~= 0.003706`
- worst-cell `P_p95 ~= 0.005741`

Because screen-plane coordinates were analytic in that experiment, P error is depth-dominated.

M4 epsilon is RMS, not p95. For an independent Gaussian-like ell=0 field, `abs(delta) p95 ~= 1.96 * epsilon`, making the M4 coarse ALL8 endpoints roughly:

- epsilon=.0015 -> abs-p95 ~= .00294
- epsilon=.0030 -> abs-p95 ~= .00588

The old scratch-fit residual scale therefore lies inside/near the coarse M4 bracket rather than an order of magnitude outside it.

This comparison is **not qualification**:
- the historical learner is fit-only, not family-disjoint;
- its residual distribution is not the M4 synthetic generator;
- p95 and RMS are different operators;
- real cross-view residual covariance is not captured by this scalar comparison.

It does show that a sacrificial family-disjoint IRIS residual pilot is high-value before expensive capacity work.

## Thin-structure caution

A current exploratory ONE/.012 source-anchor diagnostic did **not** show disproportionate D2-support loss for `SUBPATCH_THIN` anchors versus broad anchors.

That does not close the concern that a structure visible in only one or two views may be diluted or mishandled:
- self-support remains present by construction;
- the current proxy aggregates at family level;
- view-unique thin-structure correctness is not directly scored.

Therefore ONE is not used as the IRIS acceptance column, and a future unique-view thin-structure diagnostic remains required.

## Next scientific gates

1. **M4R ALL8 boundary refinement** — tighten the `.0015 .. .003` RMS bracket without rewriting M4.
2. **Sacrificial family-disjoint IRIS residual-scale pilot** — measure actual residual distribution in the same ray-depth units before deciding how valuable the DINO capacity ladder is.
3. **Consumer-validity interlock** — `ClosedRiggingVolume V0 -> InteriorRiggingSubstrate V0 -> Geppetto G0 -> Compiler -> Arachne A0 -> deformation/proof`.
4. Only then freeze a consumer-qualified product-level depth-error region and enter foundation-prior selection.

## Firewalls

- M4 cell list and decision operator remain immutable.
- M4R is a new versioned follow-up, not a post-hoc modification of M4.
- Proxy27 is historical PASS evidence but was not reused for M4 tuning.
- DEV32 remains closed.
- No product-safe IRIS claim is authorized by M4.
