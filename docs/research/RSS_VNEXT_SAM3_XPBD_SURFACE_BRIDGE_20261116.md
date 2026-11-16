# RSS.vNext SAM 3 -> explicit XPBD surface authoring (2026-11-16)

Scope: Source-observed 3D segmentation candidate, canonical face reindex, pin selection,
editable material policy. No model/trainer/RigProgram change. No dataset writes.

## Verified V4 experimental report

V4 report schema: BUSTED.KnightSAM3.MultiviewFaceEvidence.v4.
Source mesh SHA: c8e28b1f613d32e101101354a0cf86ed33de861221ec6c1a47cd2a9b6f0d8852.
Its recorded eight cameras are FROZEN_8_OBSERVED_KNIGHT_VIEWS_NOT_MODEL_CONSTRUCTION.

- Cape: 83 two-view-supported face IDs, 3 also contradicted => 80 safe candidates.
- Sword: 21 two-view-supported face IDs, 0 contradicted.
- Armor: 85 two-view-supported face IDs, 60 contradicted => 25 safe candidates.

These are SOURCE face indices, NOT canonical mesh addresses. The source Knight
demo may be different from the product demo. Never silently remap by face number.

## Compiler / Mechanics contract

File: compiler/realsas_compiler_core/mechanics_sam3_xpbd_bridge_v1.py

1. Rebuild two-view support from recorded per-view MASK_SELECTED positive faces.
2. Reject reports claiming model-generated six views as independent cameras.
3. Subtract all contradicted source faces.
4. Require explicit source-mesh-SHA -> canonical-mesh triangle reindex. The bridge
   verifies identity and canonical triangle membership, but cannot independently
   prove the caller's mapping: it must come from the original source mesh.
5. Refuse disconnected surfaces and overconstrained cut boundaries. Do not pin
   a whole inferred cape boundary just to satisfy the XPBD seam court.
6. Select limited pins using explicitly selected canonical joints, actual skin
   weights and boundary locality. An existing RigProgram attachment anchor can
   move the selection point away from a bone head. Missing attachment evidence
   is not inferred from SAM 3.
7. Require an explicit authoring policy: LIGHT_DRAPE, HEAVY_DRAPE or
   STRUCTURED_FABRIC, and scene_units_per_meter calibration. These are tunable
   material starting points, NOT recovered constitutive material measurements.
8. Call the EXISTING canonical XPBD domain and runtime; do not duplicate it.
   Explicit editor entry point:
   ControllablePuppetSessionV1.enable_sam3_assisted_xpbd_surface(...).

Authoring example needs a real source reindex (never fabricate values):

    from compiler.realsas_compiler_core.mechanics_sam3_xpbd_bridge_v1 import (
        SAM3CanonicalFaceBindingV1, SAM3XPBDAuthoringRequestV1)
    binding = SAM3CanonicalFaceBindingV1(
        source_mesh_sha256=report["source_mesh_sha256"],
        canonical_mesh_lineage_hash=session.bundle.mesh.mesh_lineage_hash,
        indexed_faces=INDEPENDENTLY_VERIFIED_SOURCE_FACE_REINDEX,
    )
    request = SAM3XPBDAuthoringRequestV1(
        concept="cape", observation_scope="ORIGINAL_FRONT_BACK",
        anchor_joint_ids=EDITOR_SELECTED_JOINT_IDS,
        attachment_anchor_ids=EDITOR_SELECTED_RIGPROGRAM_ANCHOR_IDS,
        material_profile="LIGHT_DRAPE",
        scene_units_per_meter=EDITOR_CALIBRATED_UNIT_SCALE,
        user_approved_surface_dynamics=True,
    )
    audit = session.enable_sam3_assisted_xpbd_surface(
        report=report, source_face_binding=binding,
        authoring_request=request)

## Proof boundary

Small constructed mesh tests validate source evidence filtering, provenance,
pin math, parameter choice and canonical XPBD admission. They do NOT prove the
actual frozen Knight source-face map, cloth's physical constants or product-quality
flutter. For real release, run the SAME source animation and SAME canonical mesh
with existing vs proposed pins, comparing stretch, area, collisions, temporal
flutter and visual surface continuity. ABSTAIN on missing source reindex or
unsupported anchors rather than treating segmentation as physical truth.
