# IRIS P-V5 R256 — DEV32 zero-shot family generalization probe V1

**Date:** 2026-08-25  
**Status:** `FROZEN_BEFORE_DEV_RASTER_OR_GEOMETRY_OPEN`  
**Purpose:** diagnostic zero-shot family/domain generalization after exact 8×2 fit closure; not final product qualification.

## Fixed model authority
No optimizer is authorized.
- checkpoint label: `EXACT_CONT_1024`
- total optimizer steps: `7168`
- checkpoint SHA-256: `672a92030ce1a62acd8228791eb92c7af93c34fa36ac1d5866153f38ef8708de`
- training authority: exact 8×2 continuation PASS, 16/16, worst P95 `0.004825880285352466`.

Model weights are frozen before DEV evidence is opened.

## Corpus split authority
Canonical production selection has 3993 non-QA assets:
- FIT 2981
- TUNE 318
- CAL 251
- DEV 274
- EXTERNAL_HOLDOUT 169

Builder split assignment is deterministic/hash-based and post-corpus audit recomputed it with zero mismatches.

This probe opens **DEV only**. TUNE/CAL/EXTERNAL_HOLDOUT remain unopened by this gate. `EXTERNAL_HOLDOUT` is explicitly preserved for later final qualification.

## Membership
Membership file: `P_V5_R256_DEV32_ZERO_SHOT_MEMBERSHIP_V1.json`  
Canonical JSON SHA-256: `afc20747a6154ac514f3c791ef03496b9e71f99dd7727b7804518e77767b157d`

Eligible pool:
- split = DEV
- IRIS capability = true
- `single_pose_core_eligible = true`

DEV source census from frozen master ledger:
- Objaverse linked originals: 255
- Quaternius CC0: 16
- KayKit CC0: 3

Membership size = 32 assets × 2 styles = 64 cells.

Selection rule fixed before any selected DEV image/geometry evaluation:
1. quotas = 26 Objaverse, 3 Quaternius, 3 KayKit;
2. all three KayKit DEV assets are included;
3. within each source, rank ascending by `SHA256("PV5_R256_DEV32_ZERO_SHOT_V1|" + canonical_asset_id)`;
4. take the fixed source quota;
5. no content/geometry/difficulty inspection may alter membership.

This is source-stratified diagnostic sampling, not an estimate of the natural source mixture. Report each source separately.

## Input/truth contract
Unchanged from P-V5 R256:
- neutral pose × 8 ordered views;
- canonical 512 RGBA derivative resized by PIL bilinear to 256;
- no augmentation;
- no camera JSON;
- yaw = `[0,45,...,315]`;
- `sheet_half_extent` measured from native 1024 alpha support;
- 4096 visible raster-authority rows per view;
- exact same deterministic sampling seed: `SHA256(asset_id + "|pv5-depth|" + view)`;
- truth P reconstructed from canonical geometry + raster triangle/barycentric authority;
- same truth loci shared across cel_clean and ink_cel.

## Evaluation
Fixed checkpoint only; scientific optimizer steps = 0.

Per cell:
- P mean / P50 / P90 / P95 / max;
- PASS iff `P95 <= 0.005`.

Report:
- 64 cell metrics;
- asset both-style pass;
- aggregate P metrics;
- source-stratified pass count/rate and P95 distribution;
- overall worst cell.

## Decision
`P_V5_R256_DEV32_ZERO_SHOT_STRICT_PASS` iff **64/64 cells individually have P95 <= 0.005**.

Otherwise:
`P_V5_R256_DEV32_ZERO_SHOT_NOT_CERTIFIED`.

A NOT_CERTIFIED result is a generalization diagnostic, not automatic representation/architecture failure. Localize by source/family/style and decide whether larger FIT-scale training is required before any architectural intervention.

If strict PASS:
- do not open EXTERNAL_HOLDOUT yet;
- next freeze a larger/full DEV evaluation or the next scale gate under an explicit prereg.

## Firewalls
- optimizer steps = 0;
- no checkpoint selection on DEV;
- no fine-tuning on DEV;
- no TUNE/CAL/EXTERNAL_HOLDOUT access;
- no PatchMatch;
- no architecture change;
- N/U/Z remain outside this P-only gate.
