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
- no real-family repair constants.

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

Ongoing Codec/harness changes invalidate promotion. Family selection remains blocked.

## Gate 3 — Arachne / SkinFieldCodec

**Status:** `FAIL_A0 — CLEAN SHARP WORST-ROW STABILITY SEAM`

Records:
- preregistration: `canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`
- V1 failure/correction: `canonical/ARACHNE_CODEC_BEHAVIORAL_FAILURE_20260903.md`
- first clean Bound V2 run: `canonical/ARACHNE_CODEC_BOUND_V2_FIRST_RUN_20260903.md`

Frozen authority chain:
`Codec A0 -> frozen Codec -> Arachne A1 -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`

### Independent source bug closed

Historical per-class active-weighted CE made exact teacher W non-stationary.

Repairs:
- `f0fe52ab625695d46bed7007acba39fe4cdfb248` — row-scalar active emphasis;
- `9692ac12a44769212906616b6bca13861022d42c` — exact-truth stationarity regression.

Current exact-truth CE logit gradient max: approximately `4.43e-17` PASS.

### V1 harness evidence corrected

V1 failed to rebind synthetic teacher W/rest rows from numeric creation order to lexicographically sorted canonical `conditioning.surface_ids` for N>=10.

Thus historical branch/sharp representation-capacity conclusions are `INVALID / CONTAMINATED_BY_ROW_BINDING_BUG`.

Historical contaminated free-latent and pair-geometry diagnostics remain in repo for provenance but are not active authority.

### Bound V2 — first clean run

Workflow `33756424156`, job `100651837229`, `westus`:

`chain_blend_3`:
- A0 sustained PASS step `480`;
- A1 shipping/Compiler/LBS sustained PASS step `320`.

`branch_blend_4`:
- canonical row permutation `[0,1,10,11,2,3,4,5,6,7,8,9]`;
- A0 sustained PASS step `768`, p95 `0.0214583`, deformation ratio `0.00767379`;
- A1 sustained PASS step `352`, qualified p95 `0.0422459`, deformation ratio `0.0129287`.

`sharp_fork_5`:
- canonical row permutation `[0,1,10,11,12,13,14,2,3,4,5,6,7,8,9]`;
- catastrophic V1 plateau disappears;
- final A0 p95 `0.0448788` and deformation ratio `0.00790533` individually PASS;
- row p95 oscillates around frozen `0.05` ceiling;
- only one consecutive PASS remains at step `1536`;
- authoritative status `FAIL_A0`; A1 not run.

Therefore current clean seam is:

`A0_OBJECTIVE_WORST_ROW_STABILITY != A0_P95_ACCEPTANCE_STABILITY`

It is not supported to claim sharp field representational incapacity.

### Current causal test

ID-bound A/B diagnostic now compares, under identical seed/model/LR/WD/horizon and frozen acceptance:
- current A0 objective;
- current A0 objective + generic top-10% row-L1 tail with weight `1.0`.

This tail operator already exists in the generic A1 contract. The frozen Bound V2 panel itself is unchanged.

No Codec source repair is authorized until this causal result is observed.

## Later gates — not yet opened

After Arachne/Codec closure:
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
