# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace.

**License:** proprietary / all rights reserved. Repository access or source visibility does not grant permission to use, copy, modify, redistribute or commercialize RealSaS materials. See `LICENSE`. Direct third-party dependencies and attribution boundaries are recorded in `THIRD_PARTY_NOTICES.md`.

## Start here

For a new chat, agent, machine session, technical reviewer, or investor due-diligence pass, **do not reconstruct current truth by reading dated reports in arbitrary order**.

1. **`canonical/REHYDRATION_PACKET.md`** — compact current-context reconstruction.
2. **`canonical/FIT1_EVIDENCE_INDEX_20260909.md`** — short proof chain for the current Mage FIT1 work: exact source, prereg, hashes, terminal result, promotion and non-claims.
3. **`canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`** — machine-readable frozen Geppetto evidence identities plus external artifact locators.
4. **`CURRENT_STATE.md`** — canonical stop/go and continuation authority on `main`.
5. **`canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`** — FIT1 scientific chronology.
6. **`canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`** — mechanism implementation/test/promotion status.
7. **`canonical/LIVE_AUTHORITY_MAP.md`** — generated live branch/experiment navigation.
8. **`REPOSITORY_MAP.md`** / **`SYSTEM_INDEX.md`** — deeper source/package navigation.

## Current scientific status — 2026-09-09

**Geppetto is closed for the controlled Mage FIT1 witness and separately promoted as a frozen source.**

Current Geppetto source home:

`models/geppetto/reference_strength_v1/`

The promoted formulation reached `FIT1_TERMINAL_PASS` at optimizer step `14080` and maintained the required `48/48` full structural checks across `3072` optimizer steps. The exact checkpoint/result/qualified-skeleton hashes and source lineage are in `canonical/FIT1_EVIDENCE_INDEX_20260909.md` and are cross-bound by `canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`.

**The current bottleneck is Arachne A0 / SkinFieldCodec.** No current V7-native Arachne A1 model is promoted, and no full end-to-end `PRODUCT_PASS` claim exists yet.

This distinction is deliberate: a reviewer should be able to see both what has genuinely closed and what remains scientifically open without reading chat history.

## Product boundary

RealSaS ships an **eight-direction editable 2D/2.5D puppet**. World/camera-space geometry is mechanically useful evidence; it is not full-3D reconstruction authority.

Canonical product flow:

`8 raster observations + exact cameras -> IRIS -> deterministic GSA/RiggingSurfaceIR -> Geppetto proposal -> Compiler skeleton qualification -> SkinFieldCodec/Arachne proposal -> Compiler skin/mesh qualification -> appearance/motion -> proof/repair -> export -> runtime`

## Current learned homes

- IRIS V2 observation/foundation evidence: `models/iris/v2/`
- IRIS V3 scene-first signed-geometry head and promoted Mage FIT1 witness: `models/iris/v3/`
- Geppetto FIT1-frozen: `models/geppetto/reference_strength_v1/`
- prior Geppetto V2 provenance: `models/geppetto/v2/`
- base SkinFieldCodec source: `models/skin_field_codec/v1/` — current V7 A0 research is not yet promoted
- prior/current Arachne scaffold: `models/arachne/v2/` — no present V7-native A1 FIT1 promotion

IRIS V2 and V3 are layers of one IRIS ownership envelope: V2 preserves the promoted observation/foundation/evidence stack, while V3 is the later promoted scene-first signed-field composition used by the frozen Mage upstream witness. Deterministic GSA remains the `RiggingSurfaceIR` authority.

Models emit evidence or proposals. `compiler/realsas_compiler_core/` remains the single canonical product/identity/qualification authority.

## Direct external model dependency

The current IRIS foundation directly uses **DINOv2-S** from `facebookresearch/dinov2`, bound to exact source revision and weight identity by `models/iris/v2/dinov2_foundation_v2.py`. DINO/DINOv2 may therefore retain its real upstream name where technically required. See `THIRD_PARTY_NOTICES.md`.

Other external project names may appear in comparison, clean-room audit, bibliography, or historical scientific-lineage documents. That does **not** make those projects dependencies. RealSaS-owned model/package/class/architecture identities use RealSaS mechanism names rather than third-party branding; this is enforced by `tests/repository/test_model_branding_boundary_v1.py`.

## Evidence rule

For a scientific claim, the repository should expose the chain:

`preregistration -> frozen source/apparatus -> run/result -> content hashes -> closure -> separate promotion decision`.

The current Geppetto FIT1 chain follows exactly that pattern. Large checkpoints do not need to be committed into Git to count as evidence; their byte identity is SHA-256 bound, external Drive artifact IDs are recorded, and the source/result lineage is preserved.

Local machine-level evidence consistency check:

```bash
python -m pytest -q tests/models/test_geppetto_fit1_evidence_manifest_v1.py
```

That test requires no access to the large checkpoint bytes: it verifies the committed evidence manifest, frozen Git blob identities, checkpoint authority constants, scientific chain files, claim boundaries and external artifact hash bindings. Independent byte re-download can then verify the recorded SHA-256 identities when artifact access is granted.

## Repository governance

- `CONTRIBUTING.md` — scientific-change, promotion and validation rules.
- `SECURITY.md` — private security-reporting and secret/artifact handling policy.
- `.github/CODEOWNERS` — ownership of current authority and implementation surfaces.
- `.github/pull_request_template.md` — scope/evidence checklist.
- `.editorconfig`, `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml` — repository hygiene.
- `.python-version`, `pyproject.toml`, `requirements/` — pinned current development/CPU-CI environment; sealed experiments retain their own environment authority.
- `.github/dependabot.yml` — monthly dependency update proposals.

Current scientific/mainline GitHub Actions run on the local self-hosted RealSaS runner with labels `[self-hosted, linux, x64, realsas]`; GitHub-hosted runners are not current execution authority.

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
