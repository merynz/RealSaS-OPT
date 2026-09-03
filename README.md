# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace.

## Start here

1. **`CURRENT_STATE.md`** — last canonical-main scientific state.
2. **`RESTORATION_STATE.md`** — active Compiler/runtime promotion state while the restoration branch is open.
3. **`REPOSITORY_MAP.md`** — where each class of source/evidence belongs.
4. **`canonical/README.md`** — current authority index and supersession rules.

## Product boundary

RealSaS ships an **eight-direction editable 2D/2.5D puppet**. World/camera-space geometry is mechanically useful evidence; it is not full-3D reconstruction authority.

Canonical product flow:

`IRIS -> GeometricSubstrate -> Geppetto proposal -> Compiler qualification -> Arachne proposal -> Compiler skin/mesh/appearance/motion -> proof/repair -> export -> runtime`

## Authority rule

`compiler/realsas_compiler_core/` is the single canonical product/identity/qualification authority.
Historical production knowledge may be promoted only **behind** that authority. Historical code never regains ownership merely because it is older or larger.

## Repository rule

- production authority is obvious from path;
- experiments never masquerade as production;
- historical evidence is preserved, but cannot execute by provenance alone;
- generated/archival reports do not become semantic owners;
- every promoted historical source is SHA-bound and regression-gated.

See `docs/repository/AUTHORITY_MODEL.md` and `docs/repository/STRUCTURE_POLICY.md`.
