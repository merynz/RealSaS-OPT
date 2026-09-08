# RealSaS — Arachne Mage P0 Full-Source FS1 Closure — 2026-09-08

**Status:** `PASS_DETERMINISTIC_DOUBLE_REPLAY__FS1_TARGET_SEALED__A0_STILL_BLOCKED`

## Closed gate

The preregistered repaired FS1 launcher was executed twice against the exact hash-pinned Mage inputs after commit `6e95bc0963b68b41a36b096b6f5dc8b1e8134f3d`.

Both executions produced byte-identical NPZ and manifest outputs. No projection policy, threshold, tie-break, deformation-surface eligibility rule, bridge identity mapping, or topology was changed after output inspection.

## Execution authority

- repaired launcher Git blob: `da3f53516f2dfd066e60c955df2908e26e7d5c0d`
- unchanged FS1 projection implementation Git blob: `4e5e54e893f145a1e57f76ee8a5751b38fceca0f`
- repaired JointIdentityBridge file SHA-256: `3a2f4d586a14fda2f714d24e7f3e83b5c1a830f8ee41ac3f18921b32cf9384f8`
- scene-first GSA dependency Git blob: `a55e431faf0680e0d7f2920a85fda6341d45117d`
- qualified skeleton file SHA-256: `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`
- normalized source SHA-256: `528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f`
- zero surface SHA-256: `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`
- historical body teacher SHA-256: `ac2c37571b77fa2900cc042e36f74df111432b94fb21acfb54694d5b6f962e67`

## Source deformation authority

Full normalized source:

- vertices: `5321`
- faces: `5763`
- exact dense skin: `5321 × 41`
- zero selected-22-control skin vertices: `42`
- excluded zero-skin faces: `80`
- eligible skin-supported faces: `5683`
- non-bridge source-control mass L1: `0.0`

Teacher/source geometry remains training/evaluator-only provenance and is not a product inference dependency.

## FS1 target seal

- target shape: `950 × 22`
- target dtype: `float32`
- teacher W content SHA-256: `7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d`
- target NPZ SHA-256: `7154f5ad98b446c0019c472fd91863b5843b98846cdb38b38e4f78ba7fb5b93f`
- S/G/W binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`
- manifest SHA-256: `9275d0347c2c58b2fc90b0beb45530260580369dec6105623ff0c19bb1e417aa`

The NPZ is deterministic and regenerable from the committed launcher plus pinned inputs. The repository seal records its exact content hash; the Git connector did not provide a reliable binary-file transfer path in this session, so the NPZ itself is not asserted as a committed repository blob.

## Result inventory

- observed rows: `897`
- completed rows: `53`
- distance-low-confidence rows: `16`
- local semantic ambiguity rows: `29`
- flagged union rows: `39`
- nearest candidate changed rows: `13`

Confidence:

- HIGH: `916`
- MEDIUM: `8`
- MEDIUM_LOW: `10`
- LOW: `16`

Projection modes:

- NEAREST3D_STABLE: `911`
- MULTIVIEW_VALIDATED_NEAREST: `5`
- MULTIVIEW_DISAMBIGUATED: `8`
- SINGLE_VIEW_VALIDATED_NEAREST: `5`
- SINGLE_VIEW_LOCAL_TIEBREAK: `5`
- NEAREST3D_LOW_CONF_FALLBACK: `16`

Validation:

- finite: `TRUE`
- negative weights: `0`
- max float32 simplex error: `4.190951585769653e-08`
- surface IDs canonical sorted: `TRUE`
- joint IDs canonical sorted: `TRUE`

## Historical regression

The original body-only geometric foundation remains reproduced:

- body vertices/faces: `3348 / 4029`
- distance-low-confidence rows: `16`
- local-semantic-ambiguity rows: `24`
- stable rows: `915`
- nearest mean: `0.009367747137483607`
- nearest p95: `0.03069111655845138`
- nearest p99: `0.06860541512023098`
- nearest max: `0.09476343516885569`

## Authorization

- `A0_OPTIMIZER_AUTHORIZED = FALSE`
- `A1_OPTIMIZER_AUTHORIZED = FALSE`
- `MAIN_SCIENTIFIC_A0_OPTIMIZER_STEPS = 0`

Next gate:

`REBUILD_CONDITIONING_CACHE → REBIND_A0_PREREG_AND_RUNNER → CPU/RESUME/TERMINAL_IDEMPOTENCE_REGRESSION → SELF_CONTAINED_CUDA → MAIN_A0`
