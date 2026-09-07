# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace.

## Start here — continuation / context rehydration

For a new chat, agent, machine session, or resumed research thread, **do not reconstruct current truth by reading reports in arbitrary order**.

1. **`canonical/REHYDRATION_PACKET.md`** — compact 2–5 minute current-context reconstruction view. Generated/cache; not independent authority.
2. **`CURRENT_STATE.md`** — canonical stop/go and continuation authority on `main`.
3. **`canonical/LIVE_AUTHORITY_MAP.md`** — generated live branch/experiment navigation. If stale/missing, run `python tools/render_authority_map.py`.
4. **`canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`** — mechanism implementation vs test vs canonical status.
5. **`canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`** — exact gate semantics, including what each experiment does **not** prove.
6. **`REPOSITORY_MAP.md`** / **`SYSTEM_INDEX.md`** — deeper source/package navigation.

Machine continuity state lives in:

- `canonical/CONTEXT_STATE_V1.json`
- `canonical/AUTHORITY_MAP_V1.json`

The generated views can always be reconstructed from those files + live repository refs.

## Important supersession note

`RESTORATION_STATE.md` is preserved restoration-era evidence. It is **not current continuation authority** and may contain historically correct stop/go statements that have since been superseded. Always start from `CURRENT_STATE.md` / the rehydration packet.

## Product boundary

RealSaS ships an **eight-direction editable 2D/2.5D puppet**. World/camera-space geometry is mechanically useful evidence; it is not full-3D reconstruction authority.

Canonical product flow:

`IRIS -> GSA/RiggingSurfaceIR -> Geppetto proposal -> Compiler qualification -> Arachne proposal -> Compiler skin/mesh/appearance/motion -> proof/repair -> export -> runtime`

## Authority rule

`compiler/realsas_compiler_core/` is the single canonical product/identity/qualification authority.
Historical production knowledge may be promoted only **behind** that authority. Historical code never regains ownership merely because it is older or larger.

## Repository rule

- production authority is obvious from path;
- experiments never masquerade as production;
- historical evidence is preserved, but cannot execute by provenance alone;
- generated/archival reports do not become semantic owners;
- every promoted historical source is SHA-bound and regression-gated;
- source existence != mechanism test;
- mechanism test != full-formulation verdict;
- FIT1 witness success != generalization evidence.

See `docs/repository/AUTHORITY_MODEL.md` and `docs/repository/STRUCTURE_POLICY.md`.
