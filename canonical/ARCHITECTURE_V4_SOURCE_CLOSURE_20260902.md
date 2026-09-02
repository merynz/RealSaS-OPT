# RealSaS — Architecture V4 Source Closure — 2026-09-02

**Status:** `SOURCE_CONTRACT_IMPLEMENTED__ONTOLOGY_CORRECTED__TERMINOLOGY_AUDITED__REPOSITORY_AUTHORITY_ALIGNED__FINAL_HEAD_CI_PASS__READY_TO_MERGE`

This closure records the source-level implementation of the final architecture audit decisions and the explicit correction of the 3D-equivalent/product boundary before merge.

## Implemented source contract

- `compiler/realsas_compiler_core/v4_types.py`
- `compiler/realsas_compiler_core/v4.py`
- V4-compatible `rig.py`
- support-admission-corrected `surface.py`
- V4-aware `bundle_routes.py`
- V4 facade exposure in `api.py`
- V4 authority reflected in root `README.md` and `CURRENT_STATE.md`

## Binding ontology

`3D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`.

World/camera-space depth, P, local geometry and correspondence are mechanical evidence/conditioning carriers. The shipping product is `DIRECTIONAL_2D_2P5D_PUPPET`.

The source contract explicitly carries:

- `mechanical_equivalence_class = THREE_D_EQUIVALENT_MECHANICS`;
- `representation_class = DIRECTIONAL_2D_2P5D_PUPPET`;
- `full_3d_reconstruction_authority = false`;
- puppet-local motion keys: `translation_xy`, `rotation_deg`, `scale_xy`, `depth_offset`;
- no quaternion/3D rigid-transform requirement in canonical puppet motion.

Terminology audit rule: `reconstruction` may describe a representation/field reconstruction test such as SkinFieldCodec reconstruction, but never a hidden claim that the RealSaS product reconstructs the unique full 3D character.

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

## CI evidence

Final corrected ontology head passed:

- `architecture-v4-contract`: PASS;
- `consumer-interlock-v0`: PASS;
- `consumer-coupling-probe-v1`: PASS;
- `mwb0-typed-seam-v1`: PASS;
- `mwb1-identity-subset-v1`: PASS.

The architecture regression explicitly checks that canonical puppet motion has no quaternion/3D rigid-transform fields and that full-3D reconstruction authority cannot be promoted into V4 product/runtime.

## Compatibility

Legacy `CanonicalPuppetGraph.v1`, `CanonicalPuppetGraph.v2`, `ProofFrame` and existing MWB contracts remain executable for historical/regression consumers. V4 composition authority terminates in `CanonicalPuppetGraph.v3`.

## Immediate next milestone

`SINGLE_FAMILY_E2E_FIT_V1`.

No generalization work is authorized before one real family produces a proven animated puppet.
