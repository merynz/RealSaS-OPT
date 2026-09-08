# RealSaS — Arachne Mage P0 Full-Source Supersession — 2026-09-08

**Status:** `A0_BLOCKED__FS1_PREREGISTERED__COMMITTED_BUILDER_EXECUTION_REQUIRED`

## Decision

A pre-main-A0 audit found that P0 V1 projected from `teacher_truth.npz` with `3348` body-only vertices while the normalized Mage source contains `5321` vertices / `5763` faces and accessory/attachment geometry carrying real deformation controls. P0 V1 remains scientific history and is retired as optimizer authority. No main scientific A0 optimizer step occurred before this supersession.

The full normalized source is necessary provenance, but the later source-skin audit sharpened the deformation-teacher boundary: not every normalized face is eligible teacher deformation authority.

## Why this matters

The body-only teacher omitted real deformation surfaces:

- source control `8` is `handslot.l`; Spellbook / Spellbook_open carry that control,
- source control `13` is `handslot.r`; 1H_Wand / 2H_Staff carry that control.

A full-source nearest-projection diagnostic changed `9 / 950` GSA rows relative to the body-only target. That count remains diagnostic evidence only; it is **not** a final FS1 change-count claim.

Duplicate source coordinates were separately checked and did not explain the discrepancy: duplicate-coordinate skin rows had zero L1 disagreement. The issue was omitted accessory/attachment surface authority, not seam duplication.

## FS1 deformation-surface correction

The normalized source contains `5321` vertices / `5763` faces and a dense `5321 x 41` skin field. Exact 41→22 bridge projection shows:

- `42` vertices have zero selected-22-control skin mass,
- those vertices form `80` faces,
- therefore the deformation-supported teacher surface contains `5683` eligible faces.

Those zero-skin faces are valid normalized/render provenance but cannot be selected as Arachne deformation teacher truth. FS1 therefore projects only onto faces for which all three vertices carry selected-22-control mass above the preregistered epsilon `1e-8`.

This is a provenance/mechanical eligibility rule, not an output-dependent accuracy repair.

## Frozen FS1 projection contract

Formal preregistration:

`canonical/ARACHNE_MAGE_P0_FULL_SOURCE_FS1_PREREG_20260908.json`

Committed builder:

`experiments/arachne_skintokens_fit1_20260908/build_arachne_mage_p0_full_source_fs1.py`

Frozen projection chain:

```text
full normalized source authority (5321 vertices / 5763 faces)
+ exact dense 41-column source skin
+ exact JointIdentityBridge 41→22
→ exclude zero-selected-skin deformation-ineligible faces
→ 5683 skin-supported source faces
+ exact shipping 950-node GSA
+ exact QualifiedSkeletonIR
→ exact closest top-8 candidates
→ frozen distance/local-semantic ambiguity gates
→ support-view continuous orthographic front-most ray on exact product cameras
→ hybrid vote / nearest fallback
→ FS1 950×22 target
→ confidence inventory + hashes + S/G/W binding seal
```

The numeric P0 thresholds are preserved (`0.05`, `+0.005`, local skin-L1 `0.2`, support-view vote-L1 `0.15`, multiview votes `>=2`). The historical temporary raster sampler and old P0 binary were not retained in repository/Drive, so old mode counts are not FS1 acceptance targets. FS1 freezes an explicit continuous-ray sampling contract before authoritative output sealing.

Historical geometric-foundation regression remains binding: body teacher `3348` vertices / `4029` faces, `16` distance-low-confidence rows, `24` local-semantic-ambiguity rows, and `915` stable rows, plus the sealed nearest-distance telemetry recorded in the FS1 preregistration.

## Retired fingerprints

These remain historical/reproducibility evidence and must not authorize optimization:

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

Teacher/source geometry remains training/evaluator-only provenance and is not a shipping inference input.

## Execution interlock

`experiments/arachne_skintokens_fit1_20260908/run_arachne_mage_a0_v1.py` remains a fail-closed authority-block stub. `--preflight-only`, resume and main training remain blocked until FS1 target/binding/cache authority is sealed and the A0 prereg/runner is explicitly rebound.

Machine-readable authority:

`canonical/ARACHNE_MAGE_A0_AUTHORITY_BLOCK_V2_20260908.json`
