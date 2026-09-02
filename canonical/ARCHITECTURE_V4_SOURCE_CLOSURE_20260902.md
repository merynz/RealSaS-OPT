# RealSaS — Architecture V4 Source Closure — 2026-09-02

**Status:** `SOURCE_CONTRACT_IMPLEMENTED__ONTOLOGY_CORRECTED__PENDING_SECOND_CI_AND_MERGE`

This closure records the source-level implementation of the final architecture audit decisions and the explicit correction of the 3D-equivalent/product boundary before merge.

## Implemented source contract

- `compiler/realsas_compiler_core/v4_types.py`
- `compiler/realsas_compiler_core/v4.py`
- V4-compatible `rig.py`
- support-admission-corrected `surface.py`
- V4-aware `bundle_routes.py`
- V4 facade exposure in `api.py`

## Binding ontology

`3D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`.

World/camera-space depth, P, local geometry and correspondence are mechanical evidence/conditioning carriers. The shipping product is `DIRECTIONAL_2D_2P5D_PUPPET`.

The source contract explicitly carries:

- `mechanical_equivalence_class = THREE_D_EQUIVALENT_MECHANICS`;
- `representation_class = DIRECTIONAL_2D_2P5D_PUPPET`;
- `full_3d_reconstruction_authority = false`;
- puppet-local motion keys: `translation_xy`, `rotation_deg`, `scale_xy`, `depth_offset`;
- no quaternion/3D rigid-transform requirement in canonical puppet motion.

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
- support=False cannot influence fused surface nodes;
- full 3D reconstruction authority is fail-closed forbidden in V4 product/runtime.

## Compatibility

Legacy `CanonicalPuppetGraph.v1`, `CanonicalPuppetGraph.v2`, `ProofFrame` and existing MWB contracts remain executable for historical/regression consumers. V4 composition authority terminates in `CanonicalPuppetGraph.v3`.

## Immediate next milestone

`SINGLE_FAMILY_E2E_FIT_V1`.

No generalization work is authorized before one real family produces a proven animated puppet.
