# RealSaS — Arachne Mage Joint Identity Bridge V1 — 2026-09-08

**Status:** `PASS_EXACT_LINEAGE_BRIDGE__GEOMETRY_DIAGNOSTIC_ONLY`

## Decision

The Mage Arachne teacher lane MUST NOT assign the 22 qualified joints to teacher skin columns by nearest-position or Hungarian matching.

Identity authority is instead the existing Geppetto FIT1 target/proposal/Compiler lineage:

`teacher source control -> Geppetto mechanical-core target row -> Geppetto generation index/proposal_id -> Compiler source_proposal_id -> canonical_joint_id`.

Teacher source indices are training-target provenance only; they were not learner inputs and are not a product inference dependency.

## Exact closure

Rebuilding the frozen Mage mechanical-core target from `normalized.npz` reproduces sealed target SHA-256:

`0b5a25c877116de60b710b7bb2a7848f30988e1622e2eda8084cad21c8ca23c9`.

The target contains 22 controls. Its exact teacher-source control order by content-serialized generation row is:

`[1, 19, 20, 21, 22, 2, 3, 9, 10, 11, 12, 13, 14, 4, 5, 6, 7, 8, 15, 16, 17, 18]`.

The final Geppetto proposal contains exactly 22 native proposal joints with:

- proposal IDs `P:GRS:0000 ... P:GRS:0021`;
- `generation_index_internal_only = 0 ... 21`, exact and complete;
- no teacher feedback.

The final Compiler-qualified skeleton contains exactly the same 22 `source_proposal_id` values and therefore gives a lossless proposal-to-canonical identity bridge.

Qualified parent topology translated back through `source_proposal_id` matches the frozen target parent serialization **22/22 exactly**.

## Geometry is independent validation only

Position agreement is not used to decide identity.

Final Geppetto proposal vs corresponding frozen target row:

- mean world error: `0.011942381729392954`
- p95: `0.022484874666476`
- max: `0.02447922177607631`

These numbers independently agree with the previously observed diagnostic geometry matching, but no geometric assignment is authoritative.

## Product architecture consequence

RigAnything preserves joint identity by carrying contextual generated-joint tokens into its skinning head. RealSaS preserves the equivalent identity continuity across the Compiler boundary using explicit lineage:

`proposal_id -> QualifiedJoint.source_proposal_id -> canonical_joint_id`.

Arachne remains conditioned on explicit product IR rather than hidden Geppetto features. This preserves editability and allows Arachne to be rerun after a qualified skeleton edit.

## Frozen rules

- `TEACHER_41_TO_22_SELECTION = SKIN_SUPPORTED_MECHANICAL_CORE_RULE`
- `TEACHER_22_TO_QUALIFIED_22_IDENTITY = EXACT_LINEAGE_BRIDGE`
- `POSITION_MATCHING_AS_IDENTITY_AUTHORITY = FORBIDDEN`
- `HUNGARIAN_MATCHING_AS_IDENTITY_AUTHORITY = FORBIDDEN`
- `TEACHER_SOURCE_INDEX_AS_LEARNER_INPUT = FORBIDDEN`
- `TEACHER_SOURCE_INDEX_AS_TARGET_PROVENANCE = ALLOWED`
- `PRODUCT_INFERENCE_DEPENDENCY = FALSE`
- `COMPILER_CANONICAL_ID_AUTHORITY = UNCHANGED`

## Next gate

`SUPPORT_VIEW_RASTER_BINDING_TO_TEACHER_ZBUFFER__MULTIVIEW_CONSISTENCY__NEAREST_3D_FALLBACK_FOR_COMPLETED_ONLY`

This bridge closes only joint/skin-column identity. It does not yet seal the 950-surface-node teacher skin field.
