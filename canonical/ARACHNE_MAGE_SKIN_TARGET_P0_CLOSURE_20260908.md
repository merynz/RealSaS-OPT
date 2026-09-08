# RealSaS — Arachne Mage Skin Target P0 Closure — 2026-09-08

**Status:** `P0_PASS__TARGET_HASH_SEALED__LOW_CONFIDENCE_ROWS_EXPLICIT__A0_NOT_AUTHORIZED`

## Scope

P0 constructs the training/evaluation-only Mage skin field on the exact current product consumer boundary:

`real IRIS/GSA RiggingSurfaceIR (950 nodes) + Geppetto/Compiler QualifiedSkeletonIR (22 joints)`.

No learned Arachne or codec optimizer step is part of this closure.

## Joint identity authority

Teacher source control / skin-column identity is not inferred geometrically.

Authority is the sealed bridge:

`teacher mechanical-core target row -> Geppetto generation index/proposal_id -> QualifiedJoint.source_proposal_id -> canonical_joint_id`.

The 41-control source rig contributes exactly 22 skin-supported mechanical controls. Position/Hungarian matching is diagnostic only.

See `canonical/ARACHNE_MAGE_JOINT_IDENTITY_BRIDGE_V1_20260908.md`.

## Surface projection finding

The initial idea of using support-view z-buffer samples as the universal primary mapping was rejected after exact diagnostics.

Reason: a GSA node can be geometrically almost coincident with one teacher surface while another teacher body part is front-most in a support view. GSA visibility tolerance therefore does not imply that the corresponding teacher surface is front-most in the teacher z-buffer.

Likewise, unconditional nearest-3D projection is insufficient because a small set of nodes lies close to semantically different teacher surfaces or far from the teacher mesh.

P0 therefore uses a hybrid, fail-visible policy.

## Frozen P0 projection policy

Base candidate:

`exact closest point on teacher render mesh -> barycentric interpolation of exact dense teacher skin`.

Teacher render vertices map to dense skin authority in the same world frame with maximum vertex mismatch `6.143906154658885e-08`.

A row is flagged for additional disambiguation if either:

- nearest teacher distance exceeds `0.05` world units; or
- another of the 8 closest teacher triangles lies within nearest distance + `0.005` and its interpolated skin differs from the nearest candidate by row-L1 `> 0.2`.

For flagged observed rows, support-view teacher z-buffer samples vote for candidate fields with row-L1 threshold `0.15`.

- `>=2` agreeing views may validate or override the nearest candidate;
- one view may only break a tie when the candidate is still within nearest distance + `0.005`;
- otherwise nearest-3D remains an explicitly LOW-confidence fallback.

Support-view z-buffer is not standalone authority. Nearest-3D is not unconditional authority.

## Result

Exact projected target:

- surface rows: `950`
- canonical joints: `22`
- shape: `950 x 22`
- dtype: `float32`
- negative weights: `0`
- simplex max absolute residual: `2.220446049250313e-16` before float32 serialization normalization effects
- target content SHA-256: `c15db7b78d272ac22998071e1fb1cec4c65d7366133ef16fb72824f222c852d9`
- generated NPZ SHA-256: `f3db92194660f12de4425e015f85ac4ac995fcfc44a0272047b049ffb0d36d71`

Projection modes:

- `NEAREST3D_STABLE`: 915
- `MULTIVIEW_VALIDATED_NEAREST`: 5
- `MULTIVIEW_DISAMBIGUATED`: 7
- `SINGLE_VIEW_VALIDATED_NEAREST`: 2
- `SINGLE_VIEW_LOCAL_TIEBREAK`: 2
- `NEAREST3D_LOW_CONF_FALLBACK`: 19

Only `9 / 950` rows changed away from the nearest candidate.

Confidence classes:

- HIGH: 920
- MEDIUM: 7
- MEDIUM_LOW: 4
- LOW: 19

The LOW rows remain explicit in the target artifact. They are not silently removed or presented as exact teacher authority.

## Scientific interpretation

P0 closes a reproducible teacher projection for the current Mage witness. It does not prove that the LOW-confidence rows have exact semantic correspondence to hidden source geometry, and it does not make the teacher projection a product inference dependency.

The main scientific target for Arachne remains deformation behavior on the admitted product substrate.

## Next gate

`A0_MAGE_PREREG__CONFIDENCE_HANDLING_AND_CODEC_CEILING`

Before A0 optimizer step 1, freeze:

1. how HIGH/MEDIUM/MEDIUM_LOW/LOW rows contribute to codec loss;
2. which rows contribute to exact W reconstruction qualification;
3. full-surface deformation probes so LOW rows cannot be ignored if they cause visible deformation failure;
4. optimizer/check cadence/checkpoint/resume/terminal-stability discipline.

`A0_OPTIMIZER_AUTHORIZED = FALSE` until that prereg is committed.
