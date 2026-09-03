# RealSaS — Behavioral hardening master ledger

**Opened:** 2026-09-03  
**Canonical repository:** `merynz/RealSaS-OPT`  
**Frozen comparison base:** `7f39a846ad05f91560836026e5ef6dfbc74dc731`  
**Current hardening branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Draft integration PR:** `#25`  
**Global refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`

## Purpose

Single status entrypoint for post-freeze behavioral hardening. Detailed causal evidence remains in subsystem-specific canonical records; historical failures are preserved there and are not erased by later repairs.

## Gate 1 — Geppetto

**Status:** `PASS / CLOSED INCLUDING CURRENT R6 U1 ORACLE CONSUMER GATE`

Authorities:
- `canonical/GEPPETTO_V2_BEHAVIORAL_CLOSURE_20260903.md`;
- `canonical/R6_GEPPETTO_OBSERVATION_ORACLE_CLOSURE_20260903.md`.

Closed facts:
- optimizer -> shipping proposal -> real Compiler mechanical authority PASS;
- heterogeneous behavioral panel + independent witness PASS;
- cross-region replay PASS;
- teacher graph equality diagnostic only, not product authority;
- no real-family repair constants;
- R6 one-family U0 full-surface PASS;
- R6 one-family U1 observation-oracle PASS;
- R6 heterogeneous U1 `3/3 PASS`.

Heterogeneous U1 final Geppetto p95:
- `chain_blend_3`: `~0.01630`, step `96`;
- `branch_blend_4`: `~0.01456`, step `96`;
- `sharp_fork_5`: `~0.01131`, step `160`.

Scope limit: the current capsule shells retain `159/160`, `159/160`, and `160/160` full samples across eight views. Therefore current U1 closes the frozen oracle consumer gate but does **not** establish a strong material-self-occlusion/full-hidden-surface necessity claim.

## Gate 2 — IRIS privileged-input firewall

**Status:** `PASS SOURCE FIREWALL / CLOSED`

Authorities:
- `canonical/IRIS_LEAK_SCOPE_20260903.md`;
- `canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`;
- `canonical/IRIS_REPROJECTION_V2_PRIVILEGED_INPUT_CORRECTION_20260903.md`.

Verification:
- run `33751592814`, westcentralus: firewall `4/4`, combined IRIS `24/24` PASS;
- run `33751730077`, westus3: firewall `4/4`, combined IRIS `24/24` PASS.

Historical learned IRIS results under the superseded privileged-input contract remain `QUARANTINED`.

`U2_PREDICTED_IRIS_SUBSTRATE_SUFFICIENCY = UNKNOWN`; U2 has not yet been opened as an information-sufficiency claim.

## Gate 3 — SkinFieldCodec / Arachne

**Status:** `PASS / CLOSED INCLUDING CURRENT R6 U0-U1 ORACLE CONSUMER GATE`

Authorities:
- `canonical/SKIN_FIELD_CODEC_SHIPPING_CAPACITY_AND_COOLING_V1_20260903.md`;
- `canonical/ARACHNE_SHIPPING_BEHAVIORAL_CLOSURE_20260903.md`;
- `canonical/R6_ARACHNE_ORACLE_SUBSTRATE_ONE_FAMILY_RESULT_20260903.md`;
- `canonical/R6_ARACHNE_OBSERVATION_ORACLE_CLOSURE_20260903.md`.

Shipping identities:
- Codec config hash `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`;
- Arachne config hash `ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`.

Shipping behavioral closure:
- actual default Codec A0 capacity PASS on the preregistered chain/branch/sharp panel;
- stable generic cosine A0 protocol PASS;
- default Arachne A1 -> frozen qualified shipping Codec -> raw W -> Compiler -> verified LBS `3/3 PASS`;
- Compiler correction negligible relative to acceptance and raw W independently passes;
- exact cross-region replay PASS.

R6 one-family `branch_blend_4`:
- U0 A0 PASS step `704`, final p95 `~0.03907`, deformation `~0.01586`;
- U0 A1 PASS step `320`, final p95 `~0.07997`, deformation `~0.02057`;
- U1 A0 PASS step `608`, final p95 `~0.01898`, deformation `~0.00732`;
- U1 A1 PASS step `512`, final p95 `~0.05775`, deformation `~0.03149`.

R6 heterogeneous U1:
- `chain_blend_3`: A0 step `544`, p95 `~0.04466`, deformation `~0.01171`; A1 step `96`, p95 `~0.03887`, deformation `~0.01048`;
- `branch_blend_4`: A0 step `608`, p95 `~0.01898`, deformation `~0.00732`; A1 step `512`, p95 `~0.05775`, deformation `~0.03149`;
- `sharp_fork_5`: A0 step `1056`, p95 `~0.02400`, deformation `~0.00481`; A1 step `736`, p95 `~0.08822`, deformation `~0.01489`.

All three achieved exact admitted row coverage, raw + qualified PASS, three consecutive stable checks, and Compiler correction `<1e-5`.

Causal verdict:
- `SHIPPING_CODEC_CAPACITY_ON_CURRENT_U1_SHAPES = 3/3 PASS`;
- `SHIPPING_ARACHNE_ON_CURRENT_U1_SHAPES = 3/3 PASS`;
- `COMPILER_RESCUE_EXPLAINS_PASS = FALSIFIED`;
- observation-oracle substrate is sufficient for current shipping Arachne+Codec on the admitted small synthetic panel before IRIS prediction error.

Same self-occlusion scope limitation applies: current U1 visibility is near-complete, so material hidden-surface deprivation remains a separate strengthening obligation rather than an already-proven claim.

## Architecture-freeze candidate state

**Status:** `REFREEZE NOT YET AUTHORIZED / SCIENTIFIC PREREQUISITES BEING HARDENED`

Old candidate fingerprint from run `33751730186`:
`1c6878b2e1e8cbd30a055849a64c8fe68558924e0a874de2e8fffa2e24ad7575`

This is stale candidate evidence only, not a seal.

Current architecture-freeze workflow now executes as prerequisites:
- IRIS source/generic/firewall gates;
- proof source gates;
- Geppetto generic + behavioral gates;
- shipping Codec A0 and Arachne A1 gates;
- R6 Geppetto one-family U0/U1 + heterogeneous U1;
- R6 Arachne one-family U0/U1 + heterogeneous U1.

Family selection must remain blocked until an explicit new architecture freeze is produced after all authorized behavioral seams close.

## Current P0 — MWB2 direction-local mesh/skin seam

**Status:** `OPEN / BEHAVIORAL AUTHORITY NOT YET CLOSED`

Existing source:
- `compiler/realsas_compiler_core/mwb2.py`;
- `compiler/realsas_compiler_core/mwb2_skin.py`.

Existing evidence is insufficient for product closure:
- MWB0 proves typed seam/nomenclature/solver-authority invariants;
- MWB1 is explicitly a sacrificial 3-vertex/1-face identity-subset lineage + exact weight-copy baseline and explicitly makes no product mesh/deformation quality claim;
- the current synthetic complete-E2E runner uses a trivial 4-node square for all eight directions.

Important newly exposed seam:
- all current R6 U1 Geppetto/Arachne substrates had `local_relation_count = 0`;
- current MWB2 requires an `OBSERVED_SAFE_LOCAL_RELATION_COMPLEX` with enough safe relations to form nondegenerate triangular faces;
- therefore the observation-substrate -> MWB2 local-relation complex is a real downstream behavioral obligation.

No source-mesh topology, hidden completion, random connectivity, real-family tuning, or family-specific heuristic may be introduced to hide this seam.

Next work:
1. audit the existing observation-derived local-relation producer against MWB2 consumer semantics;
2. preregister a generic directional MWB behavioral panel before outputs;
3. drive actual `S/W -> build_mwb2_candidate -> qualify_mwb2_mesh -> bind_mwb2_mesh_skin -> directional deformation/coverage behavior`;
4. use controlled relation/topology/weight mutations for causal evidence;
5. only then close MWB and move to appearance/motion/proof-runtime seams.

## Deferred repository/runtime/source hygiene task

**Status:** `DEFERRED UNTIL AFTER ARCHITECTURE FREEZE`

User-approved sequence:

`behavioral architecture closure -> architecture freeze -> semi-freeze source visibility/restoration -> full repo/compiler/runtime source audit -> restore required historical source authorities non-destructively -> re-audit/refreeze if source/authority changes require it`.

No deletion or disruptive branch/path rename while an active scientific gate is open.
