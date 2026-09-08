# RealSaS — Arachne Mage P0 Full-Source Supersession — 2026-09-08

**Status:** `A0_BLOCKED__P0_V1_BODY_ONLY_TEACHER_RETIRED__FULL_SOURCE_P0_V2_REQUIRED`

## Decision

A later pre-A0 audit found that the P0 V1 teacher projection used `teacher_truth.npz` with `3348` vertices covering the body-only render subset, while the normalized source contains `5321` vertices / `5763` faces and additional attachment/accessory geometry.

Therefore the P0 V1 target is retained as scientific history but is **retired as A0 optimizer authority**. No main scientific A0 optimizer step occurred before this supersession.

## Why this matters

The omitted source geometry carries real deformation authority:

- source control `8` is `handslot.l`; Spellbook / Spellbook_open vertices carry this control,
- source control `13` is `handslot.r`; 1H_Wand / 2H_Staff vertices carry this control.

A full-source nearest-projection diagnostic changed `9 / 950` GSA rows relative to the body-only target. This count is diagnostic evidence only; it is **not** the final P0 V2 change count. The complete hybrid projection policy must be replayed deterministically over all 950 rows before a new target is sealed.

Duplicate source coordinates were separately checked in the audit and did not explain the discrepancy: duplicate-coordinate skin rows had zero L1 disagreement. The issue is omitted accessory/attachment surface authority, not seam duplication.

## Retired fingerprints

These fingerprints remain historical/reproducibility evidence and must not authorize optimization:

- teacher W content SHA-256: `c15db7b78d272ac22998071e1fb1cec4c65d7366133ef16fb72824f222c852d9`
- target NPZ SHA-256: `f3db92194660f12de4425e015f85ac4ac995fcfc44a0272047b049ffb0d36d71`
- S/G/W binding SHA-256: `cb41eb7055b8e2646628daecdd0e31dfc079d163d5f5adaaa1a92f1ca1dfb994`
- conditioning cache SHA-256: `12484afc23d5c03cbad8020266ed5b39c3201d979e78f96380e748902152be6e`

The historical executable remains recoverable at commit `1cc449be4cc6eea81f3e8e9768f857dfb9b72477`.

## Preserved authority

This correction does **not** revoke:

- exact JointIdentityBridge lineage (`41 → 22 → canonical J:*`),
- shipping surface lineage `67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb`,
- final QualifiedSkeletonIR lineage `738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306`.

Teacher/source geometry remains training/evaluator-only provenance. It is not a shipping inference input.

## Required P0 V2 chain

```text
full normalized source geometry (5321 vertices / 5763 faces)
+ exact dense 41-column source skin
+ exact JointIdentityBridge 41→22
+ exact shipping 950-node GSA
+ exact QualifiedSkeletonIR
→ deterministic hybrid projection replay over all 950 rows
→ P0 V2 950×22 target
→ confidence inventory + hashes + binding seal
```

Only after P0 V2 is sealed may we rebuild the conditioning cache, rebind A0 preregistration/runner fingerprints, rerun CPU/resume/terminal-idempotence regressions, build the self-contained CUDA notebook, and execute main A0.

## Execution interlock

`experiments/arachne_skintokens_fit1_20260908/run_arachne_mage_a0_v1.py` is intentionally replaced at this branch state by a fail-closed authority-block stub. `--preflight-only`, resume, and main training are all blocked against P0 V1.

Machine-readable authority:

`canonical/ARACHNE_MAGE_A0_AUTHORITY_BLOCK_V2_20260908.json`
