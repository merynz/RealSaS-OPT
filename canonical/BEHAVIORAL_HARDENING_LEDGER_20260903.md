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

**Status:** `PASS / CLOSED`

Primary authorities:
- cleanroom matrix: `canonical/REALSAS_RIGANYTHING_SKINTOKENS_END_TO_END_CLEANROOM_MATRIX_20260903.md`
- shipping Codec causal closure: `canonical/SKIN_FIELD_CODEC_SHIPPING_CAPACITY_AND_COOLING_V1_20260903.md`
- shipping Arachne behavioral closure: `canonical/ARACHNE_SHIPPING_BEHAVIORAL_CLOSURE_20260903.md`

Authoritative shipping chain:

`Codec A0 -> frozen qualified shipping Codec -> default shipping Arachne A1 -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

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

With constant AdamW LR `1e-3`, `chain_blend_3` and `branch_blend_4` sustained PASS but `sharp_fork_5` repeatedly entered and left the valid region. The sharp lane nevertheless reached raw/qualified p95 below `0.05`, falsifying hard representational impossibility.

Controlled full-panel A/B changed only optimizer schedule:

`constant 1e-3` vs `CosineAnnealingLR(T_max=1536, eta_min=0)`.

Stable cosine A0 PASS:

- `chain_blend_3`: step `544`, p95 `~0.01764`, deformation `~0.00631`;
- `branch_blend_4`: step `672`, p95 `~0.02839`, deformation `~0.00795`;
- `sharp_fork_5`: step `1056`, p95 `~0.02705`, deformation `~0.00508`.

Compiler correction remained on the order of `1e-7` to `1e-6`; raw and qualified W were effectively equivalent for acceptance.

A0 verdicts:

- `SHIPPING_CODEC_REPRESENTATION_BOTTLENECK = FALSIFIED`;
- `CONSTANT_LR_1E-3_AS_STABLE_SHIPPING_A0_PROTOCOL = FALSIFIED`;
- `GENERIC_COSINE_A0_PROTOCOL = PASS`;
- `COMPILER_RESCUE_EXPLAINS_PASS = FALSIFIED`.

### Shipping A1 closure

Default shipping Arachne:

- architecture: `RealSaS.ArachneCandidate.SegmentAwareJointField.v2`;
- config hash: `ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`;
- model dim `128`;
- surface encoder layers `2`;
- attention heads `4`;
- feedforward dim `384`.

For every witness the shipping Codec was independently A0-qualified with the cosine protocol and frozen before Arachne optimization.

A1 results:

- `chain_blend_3`: step `128`, final p95 `~0.01749`, deformation `~0.00626`;
- `branch_blend_4`: step `512`, final p95 `~0.09542`, deformation `~0.02462`;
- `sharp_fork_5`: step `1760`, final p95 `~0.08293`, deformation `~0.01613`.

All three sustained three consecutive A1 PASS checks. Compiler correction again remained negligible (`~1e-7` to `1e-6`).

Teacher latent equality is diagnostic only: branch/sharp latent p95 can remain around `0.16` while final W and deformation satisfy the product behavioral gate.

### Cross-region replay

Workflow run `33770002712`:

- job `100697514568`, `westcentralus`: `3 passed`;
- exact rerun job `100717194442`, `eastus`: `3 passed`;
- identical witness pass steps and final metrics.

`CROSS_REGION_DETERMINISTIC_REPLAY = PASS`.

### Gate 3 final verdict

`SHIPPING_ARACHNE_TO_FROZEN_SHIPPING_CODEC_TO_COMPILER_TO_LBS = PASS / CLOSED`.

`HYBRID_ARACHNE_LATENT_CODEC_BOUNDARY_AS_NECESSARY_INFORMATION_BOTTLENECK = FALSIFIED` on the preregistered generic synthetic panel.

This is not a real-family fit or generalization claim.

## End-to-end cleanroom audit

**Status:** `COMPLETE AS CODE MATRIX / CONSUMER CAPACITY CLOSED / SUBSTRATE INFORMATION SUFFICIENCY OPEN`

Reference pins:
- RigAnything `d03cdb21dd134fa81df6b0947522469db3f78bd2`;
- SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`.

Main result:
- RigAnything is code-specialized toward template-free continuous skeleton generation and uses direct point-token × joint-token skinning plus aggressive deterministic mesh smoothing.
- SkinTokens is code-specialized toward a dedicated high-capacity skin representation: skin-aware dense training samples, FSQ-CVAE, autoregressive skin tokens, geometry-conditioned dense decode and optional topology/voxel prior.
- No inspected common benchmark authorizes a direct empirical claim that one globally outperforms the other.
- RealSaS has credible function-level equivalents across observation substrate, skeleton proposal/qualification, skin proposal/qualification and proof.
- Geppetto+Compiler generic behavioral capacity is closed.
- shipping Codec capacity and shipping Arachne->Codec information preservation are closed on the preregistered generic panel.

### Current P0

The remaining cleanroom architecture question is information sufficiency, not learner capacity:

`U0_REFERENCE_FULL_SURFACE` vs `U1_OBSERVATION_ORACLE_SUBSTRATE`.

Interpretation remains frozen:

- U0 fail -> consumer apparatus/representation inadequate;
- U0 pass + U1 fail -> observation-limited substrate information insufficiency demonstrated;
- U1 pass -> complete hidden/full surface is not necessary for the admitted task on that gate;
- U1 pass + U2 fail -> IRIS prediction/accessibility becomes the remaining bottleneck.

No closed-mesh reconstruction objective is implied by a U1 failure. Any failure must first be localized to a missing mechanical information class.

## Later gates — not yet opened

After substrate-equivalence closes:
- MWB / mesh-weight binding semantics;
- appearance/directional raster provenance;
- motion/runtime state mutation;
- proof/runtime fail-closed lineage and causal corruption.

No downstream gate may hide an unresolved upstream behavioral failure.

## Deferred repository/runtime/source hygiene task

**Status:** `DEFERRED UNTIL AFTER ARCHITECTURE FREEZE`

User-approved sequence:

`behavioral architecture closure -> architecture freeze -> semi-freeze source visibility/restoration -> full repo/compiler/runtime source audit -> restore required historical source authorities non-destructively -> re-audit/refreeze if source/authority changes require it`.

Non-destructive goals:
- inventory branch/PR/top-level path ownership;
- define canonical branch taxonomy/naming;
- distinguish active/frozen/audit/archive references;
- create one repository structure / branch governance map;
- preserve historical commits/records;
- expose required historical Compiler/runtime source authority without automatically promoting it to executable/canonical ownership;
- no deletion;
- no disruptive branch/path renaming while a scientific gate is active.

This task is intentionally not opened while the U0/U1 substrate gate is active.
