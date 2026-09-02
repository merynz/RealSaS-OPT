# RealSaS — Architecture V4 Source Closure — 2026-09-02

**Status:** `SOURCE_CONTRACT_IMPLEMENTED_PENDING_BRANCH_CI_AND_MERGE`

This closure records the source-level implementation of the final architecture audit decisions.

## Implemented source contract

- `compiler/realsas_compiler_core/v4_types.py`
- `compiler/realsas_compiler_core/v4.py`
- V4-compatible `rig.py`
- support-admission-corrected `surface.py`
- V4-aware `bundle_routes.py`
- V4 facade exposure in `api.py`

## Core invariants

- exact 8 directional renderables;
- shared S/G/W mechanical authority;
- direction-local M/B;
- observation-derived mesh-attached appearance;
- completion cannot become S/G/W;
- forest-safe product skeleton type;
- capability policy is product state;
- proof result is sibling-derived state;
- proof bound to exact product hash;
- missing required proof => ABSTAIN;
- required-domain FAIL => FAIL;
- runtime projection requires exact PASS bundle;
- support=False cannot influence fused surface nodes.

## Compatibility

Legacy `CanonicalPuppetGraph.v1`, `CanonicalPuppetGraph.v2`, `ProofFrame` and existing MWB contracts remain executable for historical/regression consumers. V4 composition authority terminates in `CanonicalPuppetGraph.v3`.

## Immediate next milestone

`SINGLE_FAMILY_E2E_FIT_V1`.

No generalization work is authorized before one real family produces a proven animated puppet.
