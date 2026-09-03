# RealSaS — Behavioral hardening master ledger

**Opened:** 2026-09-03  
**Canonical repository:** `merynz/RealSaS-OPT`  
**Frozen comparison base:** `7f39a846ad05f91560836026e5ef6dfbc74dc731`  
**Current hardening branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Draft integration PR:** `#25`  
**Global refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`

## Purpose

This file is the single status entrypoint for the post-freeze behavioral hardening sequence. Detailed causal evidence remains in subsystem-specific canonical records; this ledger records only the current epistemic state, authoritative evidence pointers and next gate.

Historical failures are never rewritten after a later repair. `PASS` means only the proposition named by that gate.

## Gate 1 — Geppetto train/decode/Compiler behavior

**Status:** `PASS / CLOSED`

Canonical detail:
`canonical/GEPPETTO_V2_BEHAVIORAL_CLOSURE_20260903.md`

Key closure:
- generic train/decode contract mismatches were causally exposed;
- uncertainty NLL shared-latent gradient leak was confirmed and repaired;
- actual `optimize -> generate/propose -> Compiler-qualified mechanical structure` behavioral witnesses pass;
- preregistered heterogeneous panel `wave_chain_4`, `offset_star_4`, `fork_5` passes;
- independent 3-control witness passes under the current anonymous mechanical acceptance rule;
- cross-region replay `chilecentral -> westus3` passes.

No real-family constants or family-specific repair branches were introduced.

## Gate 2 — IRIS privileged-input firewall

**Status:** `PASS SOURCE FIREWALL / CLOSED`

Historical scope:
`canonical/IRIS_LEAK_SCOPE_20260903.md`

Repair closure:
`canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`

Successor correction:
`canonical/IRIS_REPROJECTION_V2_PRIVILEGED_INPUT_CORRECTION_20260903.md`

Pre-repair causal result:
- alpha path reachable;
- absolute view slot broke joint view re-enumeration equivariance;
- mask/hull path controlled learned hypothesis-domain candidates/rays.

Repaired source contract:
- learned native input is RGB-only;
- no absolute learned view-slot identity;
- camera/candidate relation enters through analytic projected geometry;
- production Q-domain is deterministic camera-only full-frame lattice;
- mask/hull remains deterministic Gate-0 diagnostic only;
- unsealed/mask/pruned/forged domains fail closed before foundation/learner execution.

Verification:
- `33751592814` westcentralus: firewall `4/4`, combined IRIS `24/24` PASS;
- `33751730077` westus3: firewall `4/4`, combined IRIS `24/24` PASS;
- architecture-freeze prerequisite run `33751730186`: expanded suite `65/65 PASS`.

Historical learned IRIS checkpoints/metrics produced under the superseded contract remain `QUARANTINED`.

## Current architecture candidate

Architecture-freeze prerequisite run `33751730186`:
- source eligibility: `PASS_SOURCE_ELIGIBLE_FOR_FREEZE`;
- generic source count: `38`;
- candidate fingerprint: `1c6878b2e1e8cbd30a055849a64c8fe68558924e0a874de2e8fffa2e24ad7575`;
- `family_selection_authorized = false`;
- only final workflow failure: expected `FAMILY_SELECTION_BLOCKED__SOURCE_CHANGED_AFTER_FREEZE`.

This fingerprint is a candidate only, not a seal.

## Gate 3 — Arachne / SkinFieldCodec behavioral seam

**Status:** `RUNNING — FIRST PREREGISTERED EXECUTION`

Preregistration:
`canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`

Frozen panel:
- `chain_blend_3`, seed `20260921`;
- `branch_blend_4`, seed `20260922`;
- `sharp_fork_5`, seed `20260923`.

Authority chain under test:

`Codec A0 -> frozen Codec -> Arachne A1 -> shipping SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS deformation`

PASS thresholds/protocol were committed before first execution and must not be relaxed after observing results.

First behavioral CI workflow launched from commit `e58988740e3274975cbfd4c7c00d66b3f1278175`.

## Later gates — not yet opened

After Arachne/Codec behavioral closure, remaining deterministic/behavioral seams will be opened one at a time rather than globally re-audited:
- MWB / mesh-weight binding semantics;
- appearance/directional raster provenance;
- motion/runtime state mutation;
- proof/runtime fail-closed lineage and causal corruption.

No downstream gate may be used to hide an unresolved upstream behavioral failure.

## Deferred repository hygiene task

**Status:** `DEFERRED UNTIL HARDENING SEQUENCE IS STABLE`

User-requested goal: make the project read as one canonical repository rather than several overlapping work streams, without deleting provenance.

Planned non-destructive scope:
- inventory branch/PR/top-level path ownership;
- define canonical branch taxonomy and naming rules;
- distinguish active, frozen, audit and archival references;
- create a single repository-structure / branch-governance map;
- preserve all historical commits/records;
- avoid branch/path renaming while a scientific gate is actively running.

No deletion is authorized by this task.
