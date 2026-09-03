# RealSaS — Behavioral hardening master ledger

**Opened:** 2026-09-03  
**Canonical repository:** `merynz/RealSaS-OPT`  
**Frozen comparison base:** `7f39a846ad05f91560836026e5ef6dfbc74dc731`  
**Current hardening branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Draft integration PR:** `#25`  
**Global refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`

## Purpose

Single status entrypoint for post-freeze behavioral hardening. Detailed causal evidence remains in subsystem records. Historical failures are preserved; later causal corrections reclassify claims rather than erasing runs.

## Gate 1 — Geppetto train/decode/Compiler behavior

**Status:** `PASS / CLOSED`

Canonical detail:
`canonical/GEPPETTO_V2_BEHAVIORAL_CLOSURE_20260903.md`

Verification:
- generic behavioral panel and independent witness PASS through actual shipping proposal / Compiler mechanical authority;
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
- `33751730077` westus3: firewall `4/4`, combined IRIS `24/24` PASS;
- architecture-freeze prerequisite run `33751730186`: `65/65 PASS` before later Codec changes.

Historical learned IRIS checkpoints/metrics under the superseded privileged-input contract remain `QUARANTINED`.

## Architecture candidate state

Old candidate fingerprint from run `33751730186`:
`1c6878b2e1e8cbd30a055849a64c8fe68558924e0a874de2e8fffa2e24ad7575`

**Status:** `STALE CANDIDATE ONLY / NOT A SEAL`

Ongoing Codec/harness changes make this fingerprint ineligible for promotion. Family selection remains blocked.

## Gate 3 — Arachne / SkinFieldCodec behavioral seam

**Status:** `BOUND_V2 RUNNING — V1 N>=10 IMPLEMENTATION EVIDENCE INVALID`

Preregistration:
`canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`

Failure/correction record:
`canonical/ARACHNE_CODEC_BEHAVIORAL_FAILURE_20260903.md`

Frozen witnesses and all scientific thresholds/protocols remain unchanged:
- `chain_blend_3`, seed `20260921`;
- `branch_blend_4`, seed `20260922`;
- `sharp_fork_5`, seed `20260923`.

Authority chain remains:

`Codec A0 -> frozen Codec -> Arachne A1 -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`

### Historical V1 observation

Run `33752571671`, job `100639284377`:
- chain full PASS;
- branch reported FAIL_A0;
- sharp reported FAIL_A0.

### Confirmed independent source bug

Historical Codec active-weighted CE changed relative targets inside a simplex row, so exact teacher W was not stationary.

Repairs:
- `f0fe52ab625695d46bed7007acba39fe4cdfb248` — active emphasis changed to teacher-only row scalar;
- `9692ac12a44769212906616b6bca13861022d42c` — exact-truth stationarity regression.

Post-repair exact teacher gradient is approximately numerical zero (`4.43e-17`). This finding remains valid independently of the panel harness issue.

### Critical V1 harness falsification

`ArachneConditioningAdapter` sorts `surface_id` lexicographically. The V1 panel generated teacher W/rest rows in numeric witness creation order and did not rebind them to `conditioning.surface_ids`.

Consequences:
- `chain_blend_3`, N=9: identity order, unaffected;
- `branch_blend_4`, N=12: conditioning order `0,1,10,11,2,...,9`;
- `sharp_fork_5`, N=15: conditioning order `0,1,10,11,12,13,14,2,...,9`.

Therefore branch/sharp V1 targets were attached to wrong canonical surface rows.

Causal replay `33756078567`, job `100650698374`, `mexicocentral`, changed only ID-axis binding while preserving seed/model/LR/WD/horizon/threshold/stability:
- branch: sustained A0 PASS at step `544`, p95 `0.0287868`, deformation ratio `0.0104174`;
- sharp: final individual PASS p95 `0.0390133`, deformation ratio `0.00743765`, but only `2` consecutive PASS checks by frozen step `1536`.

Thus:

`V1_BRANCH_SHARP_CAPACITY_CLAIMS = INVALID / CONTAMINATED_BY_ROW_BINDING_BUG`

Earlier free-latent and pair-geometry representation diagnostics used the same bad binding. Their numeric logs remain provenance, but they cannot justify source architecture changes.

### Active successor authority

Historical V1 harness is preserved unchanged:
`test_arachne_v2_behavioral_panel.py`

Corrected successor:
`test_arachne_v2_behavioral_panel_bound_v2.py`

V2 changes only canonical target-axis binding. Witness definitions, seeds, architectures, objectives, optimizer settings, horizons, thresholds and three-consecutive-PASS rule are unchanged.

Active CI now requires:
1. Codec exact-truth stationarity regression;
2. canonical row-binding cause regression;
3. full Bound V2 A0 -> A1 -> Compiler -> LBS frozen panel;
4. Arachne/Codec source gates.

No representation repair is authorized until Bound V2 produces clean evidence.

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
