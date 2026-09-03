# RealSaS — Behavioral hardening master ledger

**Opened:** 2026-09-03  
**Canonical repository:** `merynz/RealSaS-OPT`  
**Frozen comparison base:** `7f39a846ad05f91560836026e5ef6dfbc74dc731`  
**Current hardening branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Draft integration PR:** `#25`  
**Global refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`

## Purpose

Single status entrypoint for post-freeze behavioral hardening. Detailed causal evidence remains in subsystem records. Historical failures are preserved; causal corrections reclassify claims rather than erasing runs.

## Gate 1 — Geppetto

**Status:** `PASS / CLOSED`

Canonical detail: `canonical/GEPPETTO_V2_BEHAVIORAL_CLOSURE_20260903.md`

- optimize -> shipping proposal -> Compiler mechanical authority PASS;
- heterogeneous panel + independent witness PASS;
- cross-region `chilecentral -> westus3` replay PASS;
- no real-family repair constants;
- teacher graph equality remains diagnostic, not product authority.

## Gate 2 — IRIS privileged-input firewall

**Status:** `PASS SOURCE FIREWALL / CLOSED`

Records:
- `canonical/IRIS_LEAK_SCOPE_20260903.md`
- `canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`
- `canonical/IRIS_REPROJECTION_V2_PRIVILEGED_INPUT_CORRECTION_20260903.md`

Verification:
- `33751592814` westcentralus: firewall `4/4`, combined IRIS `24/24` PASS;
- `33751730077` westus3: firewall `4/4`, combined IRIS `24/24` PASS.

Historical learned IRIS results under the superseded privileged-input contract remain `QUARANTINED`.

## Architecture candidate state

Old candidate fingerprint from run `33751730186`:
`1c6878b2e1e8cbd30a055849a64c8fe68558924e0a874de2e8fffa2e24ad7575`

**Status:** `STALE CANDIDATE ONLY / NOT A SEAL`

Ongoing hardening/audit work invalidates promotion. Family selection remains blocked.

## Gate 3 — Arachne / SkinFieldCodec

**Status:** `AUDIT HOLD — SHIPPING CAPACITY UNPROVEN; TINY-CODEC FAILURE IS NOT PRODUCT FAILURE`

Primary cleanroom authority:
- `canonical/REALSAS_RIGANYTHING_SKINTOKENS_END_TO_END_CLEANROOM_MATRIX_20260903.md`

Historical records preserved:
- preregistration: `canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`
- V1 failure/correction: `canonical/ARACHNE_CODEC_BEHAVIORAL_FAILURE_20260903.md`
- first clean Bound V2 run: `canonical/ARACHNE_CODEC_BOUND_V2_FIRST_RUN_20260903.md`
- hard-tail diagnostic: `canonical/SKIN_FIELD_CODEC_A0_TAIL_DIAGNOSTIC_20260903.md`
- global-temperature diagnostic: `canonical/SKIN_FIELD_CODEC_A0_TEMPERATURE_DIAGNOSTIC_20260903.md`
- post-PASS force decomposition: `canonical/SKIN_FIELD_CODEC_A0_FORCE_DECOMPOSITION_20260903.md`
- reconstruction/cooling diagnostics: `canonical/SKIN_FIELD_CODEC_A0_RECONSTRUCTION_COOLING_DIAGNOSTICS_20260903.md`

Frozen intended authority chain:
`Codec A0 -> frozen Codec -> Arachne A1 -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`

### Independent source bug closed

Historical per-class active-weighted CE made exact teacher W non-stationary.

Repairs:
- `f0fe52ab625695d46bed7007acba39fe4cdfb248` — row-scalar active emphasis;
- `9692ac12a44769212906616b6bca13861022d42c` — exact-truth stationarity regression.

Current exact-truth CE logit gradient max was approximately `4.43e-17` PASS.

### Historical V1 harness contamination preserved

V1 failed to rebind synthetic teacher W/rest rows from numeric creation order to lexicographically sorted canonical `conditioning.surface_ids` for N>=10.

Therefore historical branch/sharp representation-capacity conclusions from that harness remain:

`INVALID / CONTAMINATED_BY_ROW_BINDING_BUG`.

The historical runs remain in the repository and are not deleted.

### Bound V2 tiny-model evidence

After correcting row binding:

- `chain_blend_3`: A0 sustained PASS and A1/Compiler/LBS PASS;
- `branch_blend_4`: A0 sustained PASS and A1/Compiler/LBS PASS;
- `sharp_fork_5`: tiny A0 repeatedly entered the frozen acceptance region but failed to hold three consecutive checks.

Subsequent generic hypotheses were tested and falsified as product repairs, including top-tail weighting, frozen temperature, weight decay, removing deformation loss, cosine cooling as a generic solution, encoder bypass, and injected 4D/10D pair-geometry residual paths.

### Critical cleanroom reclassification

The behavioral panel's helper Codec is **not shipping-sized**. It instantiates approximately:

`hidden=32, latent=8, layers=2`

while the shipping/default `SkinFieldCodecConfigV1` is:

`hidden=192, latent=64, encoder_layers=3, decoder_layers=3`.

Therefore:

`TINY_SHARP_FAIL != SHIPPING_CODEC_PRODUCT_FAIL`.

The previous status `FAIL_A0 — CLEAN SHARP NARROW REPRESENTATION/OBJECTIVE FLOOR` is preserved as a historical statement about the tiny Bound-V2 test model, but is no longer authoritative for product architecture capacity.

### Cleanroom architectural risk

Arachne consumes an explicit 10D point/control/parent-segment geometry contract and therefore has strong mechanical conditioning. The final Codec decoder receives surface features + joint features + per-joint latent rather than the full pairwise geometry tensor directly. This split may be sufficient, but its information-preservation capacity is not yet proven.

Compiler skin qualification is a strong fail-closed legality/simplex/lineage boundary, but it cannot repair semantically wrong W into correct W.

### Next authorized gates

1. **P0 shipping-sized Codec capacity smoke** using the actual default `192/64/3` architecture; sustained reconstruction + verified deformation authority.
2. **P0 full-surface-oracle vs observation-oracle mechanical ceiling** to test whether the references' visibility-independent closed-surface coverage supplies mechanically necessary information.
3. **P0/P1 Arachne -> Codec information-preservation test** at shipping dimensions.
4. Only after those pass/fail causally may a Codec/Arachne architecture repair be authorized.

No family-specific tuning, no architecture refreeze and no formal family selection are authorized while these are unresolved.

## End-to-end cleanroom audit

**Status:** `COMPLETE AS CODE MATRIX / TEST OBLIGATIONS OPEN`

Reference pins:
- RigAnything `d03cdb21dd134fa81df6b0947522469db3f78bd2`;
- SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`.

Main result:
- RigAnything is code-specialized toward template-free continuous skeleton generation and uses direct point-token × joint-token skinning plus aggressive deterministic mesh smoothing.
- SkinTokens is code-specialized toward a dedicated high-capacity skin representation: skin-aware dense training samples, FSQ-CVAE, autoregressive skin tokens, geometry-conditioned dense decode and optional topology/voxel prior.
- No inspected common benchmark authorizes a direct empirical claim that one globally outperforms the other.
- RealSaS has credible function-level equivalents across observation substrate, skeleton, skin qualification and proof, but two P0 empirical obligations remain: observation-substrate information sufficiency and shipping Arachne/Codec capacity/information preservation.

## Later gates — not yet opened

After the P0 Arachne/Codec and substrate-equivalence obligations close:
- MWB / mesh-weight binding semantics;
- appearance/directional raster provenance;
- motion/runtime state mutation;
- proof/runtime fail-closed lineage and causal corruption.

No downstream gate may hide an unresolved upstream behavioral failure.

## Deferred repository hygiene task

**Status:** `DEFERRED UNTIL HARDENING SEQUENCE IS STABLE`

User-requested non-destructive goal:
- inventory branch/PR/top-level path ownership;
- define canonical branch taxonomy/naming;
- distinguish active/frozen/audit/archive references;
- create one repository structure / branch governance map;
- preserve historical commits/records;
- no deletion;
- no disruptive branch/path renaming while a scientific gate is active.
