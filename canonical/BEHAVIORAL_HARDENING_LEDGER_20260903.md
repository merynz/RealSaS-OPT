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

**Status:** `A0 SHIPPING CODEC CAPACITY PASS / GENERIC COSINE PROTOCOL PASS / SHIPPING A1 NEXT`

Primary authorities:
- cleanroom matrix: `canonical/REALSAS_RIGANYTHING_SKINTOKENS_END_TO_END_CLEANROOM_MATRIX_20260903.md`
- shipping Codec causal closure: `canonical/SKIN_FIELD_CODEC_SHIPPING_CAPACITY_AND_COOLING_V1_20260903.md`

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

### Tiny-model evidence reclassified

The historical behavioral helper Codec used approximately `32 hidden / 8 latent / 2 encoder / 2 decoder layers`; the shipping/default Codec is `192 hidden / 64 latent / 3 encoder / 3 decoder layers`.

Therefore the historical tiny sharp failure is not product-capacity authority:

`TINY_SHARP_FAIL != SHIPPING_CODEC_PRODUCT_FAIL`.

### Shipping A0 capacity closure

The actual default shipping Codec was tested on the same preregistered three-witness panel through:

`teacher W -> shipping Codec -> raw W -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

No witness, seed or acceptance threshold was changed.

With constant AdamW LR `1e-3`, `chain_blend_3` and `branch_blend_4` sustained PASS but `sharp_fork_5` repeatedly entered and left the valid region and failed the three-consecutive criterion. The sharp lane nevertheless reached raw/qualified p95 substantially below `0.05`, falsifying hard representational impossibility.

A controlled full-panel A/B changed only optimizer schedule:

`constant 1e-3` vs `CosineAnnealingLR(T_max=1536, eta_min=0)`.

Cosine sustained PASS on all three witnesses:

- `chain_blend_3`: pass step `544`, final p95 `~0.01764`, deformation ratio `~0.00631`;
- `branch_blend_4`: pass step `672`, final p95 `~0.02839`, deformation ratio `~0.00795`;
- `sharp_fork_5`: pass step `1056`, final p95 `~0.02705`, deformation ratio `~0.00508`.

Compiler correction remained only on the order of `1e-7` to `1e-6`; raw and qualified W were effectively equivalent for the acceptance decision.

Current causal verdicts:

- `SHIPPING_CODEC_REPRESENTATION_BOTTLENECK = FALSIFIED` on the preregistered generic synthetic panel;
- `CONSTANT_LR_1E-3_AS_STABLE_SHIPPING_A0_PROTOCOL = FALSIFIED`;
- `GENERIC_COSINE_A0_PROTOCOL = PASS` on the current three-witness panel;
- `COMPILER_RESCUE_EXPLAINS_PASS = FALSIFIED`.

This is a capacity/training-protocol closure, not a real-family generalization claim.

### Remaining hybrid risk

The cleanroom concern is now narrowed to the actual learned handoff:

`shipping S + Qualified G -> default ArachneCandidateV2 -> per-joint latent -> frozen qualified shipping Codec decoder -> W`.

A0 proves the shipping Codec can represent the field when teacher W is available to its encoder. It does not yet prove that Arachne can infer an equivalent latent from product conditioning.

### Next authorized gate

**P0 shipping Arachne -> frozen shipping Codec -> Compiler -> verified LBS.**

Requirements:
- use the same preregistered three witness families and existing A1 thresholds;
- use default shipping Arachne (`model_dim=128`, `surface_encoder_layers=2`, `attention_heads=4`, `feedforward_dim=384`), not the historical tiny A1 surrogate;
- freeze a shipping Codec that has passed the generic cosine A0 protocol;
- evaluate raw decoded W and actual `propose -> qualify_skin -> QualifiedSkinIR -> verified LBS` separately;
- do not credit Compiler correction as semantic prediction quality;
- if A1 fails, localize the Arachne-to-latent seam before any architecture change.

No family-specific tuning, no architecture refreeze and no formal family selection are authorized while this is unresolved.

## End-to-end cleanroom audit

**Status:** `COMPLETE AS CODE MATRIX / TEST OBLIGATIONS OPEN`

Reference pins:
- RigAnything `d03cdb21dd134fa81df6b0947522469db3f78bd2`;
- SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`.

Main result:
- RigAnything is code-specialized toward template-free continuous skeleton generation and uses direct point-token × joint-token skinning plus aggressive deterministic mesh smoothing.
- SkinTokens is code-specialized toward a dedicated high-capacity skin representation: skin-aware dense training samples, FSQ-CVAE, autoregressive skin tokens, geometry-conditioned dense decode and optional topology/voxel prior.
- No inspected common benchmark authorizes a direct empirical claim that one globally outperforms the other.
- RealSaS has credible function-level equivalents across observation substrate, skeleton, skin qualification and proof. The shipping Codec representation-capacity question is now closed on the generic synthetic panel; remaining P0 obligations are Arachne-to-Codec information preservation and observation-substrate information sufficiency.

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
