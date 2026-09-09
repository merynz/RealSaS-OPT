# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace.

## Start here

For a new chat, agent, machine session, technical reviewer, or investor due-diligence pass, **do not reconstruct current truth by reading dated reports in arbitrary order**.

1. **`canonical/REHYDRATION_PACKET.md`** — compact current-context reconstruction.
2. **`canonical/FIT1_EVIDENCE_INDEX_20260909.md`** — short proof chain for the current Mage FIT1 work: exact source, prereg, hashes, terminal result, promotion and non-claims.
3. **`CURRENT_STATE.md`** — canonical stop/go and continuation authority on `main`.
4. **`canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`** — FIT1 scientific chronology.
5. **`canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`** — mechanism implementation/test/promotion status.
6. **`canonical/LIVE_AUTHORITY_MAP.md`** — generated live branch/experiment navigation.
7. **`REPOSITORY_MAP.md`** / **`SYSTEM_INDEX.md`** — deeper source/package navigation.

## Current scientific status — 2026-09-09

**Geppetto is closed for the controlled Mage FIT1 witness and separately promoted as a frozen source.**

Current Geppetto source home:

`models/geppetto/reference_strength_v1/`

The promoted formulation reached `FIT1_TERMINAL_PASS` at optimizer step `14080` and maintained the required `48/48` full structural checks across `3072` optimizer steps. The exact checkpoint/result/qualified-skeleton hashes and source lineage are in `canonical/FIT1_EVIDENCE_INDEX_20260909.md`.

**The current bottleneck is Arachne A0 / SkinFieldCodec.** No current V7-native Arachne A1 model is promoted, and no full end-to-end `PRODUCT_PASS` claim exists yet.

This distinction is deliberate: a reviewer should be able to see both what has genuinely closed and what remains scientifically open without reading chat history.

## Product boundary

RealSaS ships an **eight-direction editable 2D/2.5D puppet**. World/camera-space geometry is mechanically useful evidence; it is not full-3D reconstruction authority.

Canonical product flow:

`8 raster observations + exact cameras -> IRIS -> deterministic GSA/RiggingSurfaceIR -> Geppetto proposal -> Compiler skeleton qualification -> SkinFieldCodec/Arachne proposal -> Compiler skin/mesh qualification -> appearance/motion -> proof/repair -> export -> runtime`

## Current learned homes

- IRIS: `models/iris/v2/`
- Geppetto FIT1-frozen: `models/geppetto/reference_strength_v1/`
- prior Geppetto V2 provenance: `models/geppetto/v2/`
- base SkinFieldCodec source: `models/skin_field_codec/v1/` — current V7 A0 research is not yet promoted
- prior/current Arachne scaffold: `models/arachne/v2/` — no present V7-native A1 FIT1 promotion

Models emit evidence or proposals. `compiler/realsas_compiler_core/` remains the single canonical product/identity/qualification authority.

## Evidence rule

For a scientific claim, the repository should expose the chain:

`preregistration -> frozen source/apparatus -> run/result -> content hashes -> closure -> separate promotion decision`.

The current Geppetto FIT1 chain follows exactly that pattern. Large checkpoints do not need to be committed into Git to count as evidence; their byte identity is SHA-256 bound and the source/result lineage is preserved.

## Important non-claims

- FIT1 same-witness success is not unseen-family generalization.
- Geppetto FIT1 PASS is not Arachne PASS.
- SkinFieldCodec A0 PASS would not itself be Arachne A1 PASS.
- Learned skinning closure would still not automatically create `PRODUCT_PASS`; product acceptance has its own independent contract.

## Repository rule

- production/current source authority is obvious from path or explicit promotion record;
- experiments never masquerade as promoted product truth;
- historical evidence is preserved but cannot execute by provenance alone;
- source existence != mechanism test;
- mechanism test != full-formulation verdict;
- scientific PASS != automatic promotion;
- branch recency != authority;
- Compiler/runtime consumers cannot mint learned semantics that their upstream models failed to provide.

See `docs/repository/AUTHORITY_MODEL.md`, `docs/repository/STRUCTURE_POLICY.md`, and `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`.
