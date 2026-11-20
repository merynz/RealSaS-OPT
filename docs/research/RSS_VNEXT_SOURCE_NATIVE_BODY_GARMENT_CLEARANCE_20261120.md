# RSS.vNext — Source-native garment/body signed clearance on posed M* (2026-11-20)

**Status:** Optional source-qualified loss in the active seeded structural trainer, with tiny synthetic gradient and native-receipt regressions. **Not:** cloth physics, collision detection, XPBD solver, automatically inferred material parameters, native source dataset truth or ProductProof.

## Why this is necessary

A posed vertex residual alone may fit a source garment without explicitly preserving its positive separation from the body. A semantic label like "garment" cannot make a 3D penetration witness, and an open cape does **not** have a legitimate inside/outside signed-distance field.

Use the independently native-sourced **closed, oriented body shell** as the sign reference, paired with source-native garment vertex and an independently qualified point on a **body M* face** (the barycentric combination). The source pose drives joint matrices already qualified in AXIS/MIRA. This is geometric/mechanical **consequence supervision** in the same M*, not a global SDF teacher or another mesh.

## Typed source teacher / qualification

`SourceNativeLayerClearanceTeacherV1` (loss only):
- `source_asset_sha256`, `source_qualification_key`, both immutable;
- `garment_vertex_ids [K]` and `body_face_ids [K]` in the **same final TESSA M*** index namespace, source-face/barycentric reindexed, never student nearest;
- `body_face_barycentric [K,3]` nonnegative, normalized; `qualified_pose_pair_mask [1,P,K]` known sampled source pairs, UNKNOWN masked;
- independent source pose candidate `source_candidate_posed [1,P,V,3]`, valid `source_pose_mask [1,P,V]`, native pose transport and joint-order verified by the existing AXIS/MIRA teacher path.

The **trusted Foundry launch registry** SHA-pins every field, dataset manifest, source license lineage, M* topology address, native pose transport receipt and the source pose bytes. It must explicitly certify:
- body face winding outward on a source-native **closed signed body component**; an open cape does not need SDF sign but the body reference must have a verified orientation;
- independently source-native face-to-garment pair correspondence (not a guessed nearest student face);
- source rights and known positive separation for every admitted pair/pose;
- **no physical material / XPBD authority**.

The loss recomputes the signed source gap from the actual source pose, body face barycentric location and source body oriented face normal. Source pairs with a degenerate normal, missing vertex pose support or nonpositive source gap are **rejected**, not relabeled as physical negative contacts. Student geometry gets gradients through its own posed body-face barycentric coordinates and garment vertices; source normal is detached and cannot alter the forward H(x).

Normalized objective per admitted sample:
- `smooth_l1(student_signed_gap / source_scale, source_signed_gap / source_scale)`;
- plus `relu(0.5*source_signed_gap/source_scale - student_signed_gap/source_scale)^2` separation collapse barrier.

The 0.5 factor is a **declared engineering training margin**, not a physically measured stiffness/contact threshold. Its fixed coefficient lives in code and would need a new objective-policy revision if changed. Loss can influence TESSA rest vertices, predicted AXIS/MIRA skin and shared carrier continuous paths. No differentiability claim is made for discrete face topology or FSQ token argmax.

## Production trainer policy

- Default **OFF**, activated by `product_native_layer_clearance_weight > 0`.
- Requires the product seeded profile, native source rig/skin and posed-surface source loss, matching native pose transport, independent registry and intact *single M***.
- Optional per source-qualified asset under sparse mode. Absent source pair is UNKNOWN, not evidence of body/garment intersection.
- Tagged `POSED_LAYER_CLEARANCE` as an independent multi-task gradient family, not wrongly folded into semantic or Graphics.
- Checkpoint and resume identity pins the clearance coefficient and source registry digest; incompatible teacher receipts reject before optimizer update.
- Foundry handoff reports the conditional source evidence requirement. This is a **teacher-forced posed surface** target; actual native C++ XPBD post-physics clearance and visual parity still require separate source-independent acceptance and cannot be assumed.

## Evidence required before declaring training final

1. Actual licensed source character layers and independently replayed closed body winding/face correspondences, including long capes, robes, armor and nonhumanoids.
2. Full source masked coverage by pose and morphology, and held-out cloth-body gap and penetration metrics; no training-label leakage.
3. Compare baseline geometry+pose+2D against added clearance on the same source, with bounded cross-head gradient conflicts and identical compute budgets.
4. Real GPU backward/memory, native C++ post-XPBD collision and graphics parity, editor save/reopen, and final ProductProof.

Source: `models/rss_vnext/product_native_layer_clearance_v1.py`, active `models/rss_vnext/training_engine.py`, source data contract `docs/rss_vnext/BUSTED_RSSvNext_Veri_Supervision_Teslim_Sozlesmesi_2026-11-20.md`.
