# K3 original source Cycles: sealed shader replay authority amendment

2026-11-06. Original source run 42246970143 passed all K3 mathematical contracts and SHA-locked source/face/triangulation validation, but failed CLOSED before physical sample rendering at SOURCE_UNLIT_POLICY_CHANGED_alpha_policy.

Cause: original SourceRestRenderCandidate was authored under Blender 4.0.2 import with explicit IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE. Blender 4.5.14 source import reports its current material blend policy differently. Reapplying observation producer preserve_alpha_unlit would silently use Blender 4.5 *current importer state*, not the previously sealed source-replay material authority.

The original V18n source replay, however, already solves precisely this cross-Blender boundary: tools/truth_foundry/blender_source_appearance_policy_replay_v1.py::_replay_transform(materials, transform_rows) consumes the actual immutable source observation material_transform policy and does not infer opacity from the new import blend method. V18n previously passed independent original-source material parity using this function. The only permitted correction here is therefore to call this existing qualified replay function on the ORIGINAL imported shader materials with the exact sealed candidate transform rows, and validate its explicit operation/alpha_policy receipts. Any source socket/alpha drift remains FAIL. Do not override source intrinsic RGB, sample counts, original polygons, camera, paint edits, oracle thresholds or authority.

First original K3 execution (42246423433): failed source polygons already-triangulated precondition, no edit samples.
Second original K3 execution (42246970143): source face/canonical triangle preconditions now passed; material opaque replay mismatch, no edit samples.
Neither counts as a physical K3 experimental PASS or FAIL. Their logs remain immutable and source-private.

All artist owner / IRIS / TESSA / training / materialization authority remains FALSE.
