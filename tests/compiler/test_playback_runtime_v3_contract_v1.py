from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    DepthWritePolicy,
    RuntimeV3AppearancePatch,
    RuntimeV3ClipInterval,
    RuntimeV3FrameComposition,
    RuntimeV3Mesh,
    RuntimeV3PlaybackContract,
    RuntimeV3Slot,
    RuntimeV3Vertex,
    RuntimeV3VisibilityPolicy,
    TopologyClass,
    validate_frame_composition_v3,
    validate_playback_runtime_v3_contract,
)
from compiler.realsas_compiler_core.types import QualificationError


def _mesh(*, mesh_id: str, slot_id: str, attachment_id: str, kind=AttachmentKind.DEFORMABLE_BODY):
    return RuntimeV3Mesh(
        mesh_id=mesh_id,
        slot_id=slot_id,
        attachment_id=attachment_id,
        attachment_kind=kind,
        topology_class=TopologyClass.STATIC,
        vertices=(
            RuntimeV3Vertex(0.0, 0.0, 0.2, 0.0, 0.0),
            RuntimeV3Vertex(1.0, 0.0, 0.2, 1.0, 0.0),
            RuntimeV3Vertex(0.0, 1.0, 0.2, 0.0, 1.0),
        ),
        triangles=((0, 1, 2),),
    )


def _contract(*, provenance=AppearanceProvenance.DIRECT_SOURCE, allow_completion=False):
    body = _mesh(mesh_id="body", slot_id="body_slot", attachment_id="body_attachment")
    return RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        meshes=(body,),
        appearance_patches=(
            RuntimeV3AppearancePatch(
                patch_id="body_face_0",
                mesh_id="body",
                face_indices=(0,),
                provenance=provenance,
                donor_view_index=(0 if provenance != AppearanceProvenance.UNSEEN else None),
                atlas_id=("V0" if provenance != AppearanceProvenance.UNSEEN else None),
                completion_method=("explicit_test_completion" if provenance == AppearanceProvenance.COMPLETION else None),
            ),
        ),
        allow_completion=allow_completion,
    )


def test_v3_contract_is_subject_agnostic_and_deterministic():
    c = _contract()
    h1 = validate_playback_runtime_v3_contract(c)
    h2 = validate_playback_runtime_v3_contract(c)
    assert len(h1) == 64
    assert h1 == h2
    source = __import__(
        "compiler.realsas_compiler_core.playback_runtime_v3",
        fromlist=["dummy"],
    ).__file__
    text = open(source, "r", encoding="utf-8").read().lower()
    assert "mage" not in text


def test_unseen_is_explicit_and_unbound_not_fake_filled():
    c = _contract(provenance=AppearanceProvenance.UNSEEN)
    assert len(validate_playback_runtime_v3_contract(c)) == 64
    patch = c.appearance_patches[0]
    assert patch.donor_view_index is None
    assert patch.atlas_id is None


def test_completion_is_fail_closed_until_product_policy_enables_it():
    with pytest.raises(QualificationError, match="RUNTIME_V3_COMPLETION_NOT_ALLOWED_BY_PRODUCT_POLICY"):
        validate_playback_runtime_v3_contract(_contract(provenance=AppearanceProvenance.COMPLETION))
    assert len(validate_playback_runtime_v3_contract(
        _contract(provenance=AppearanceProvenance.COMPLETION, allow_completion=True)
    )) == 64


def test_body_self_occlusion_cannot_fall_back_to_draw_order():
    c = _contract()
    bad = RuntimeV3PlaybackContract(
        slots=c.slots,
        meshes=c.meshes,
        appearance_patches=c.appearance_patches,
        visibility=RuntimeV3VisibilityPolicy(
            body_depth_test=False,
            body_depth_write=DepthWritePolicy.OFF,
        ),
    )
    with pytest.raises(QualificationError, match="RUNTIME_V3_BODY_DEPTH_TEST_REQUIRED"):
        validate_playback_runtime_v3_contract(bad)


def test_appearance_authority_is_face_patch_not_unqualified_vertex_donor_mix():
    body = RuntimeV3Mesh(
        mesh_id="body",
        slot_id="body_slot",
        attachment_id="body_attachment",
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        vertices=(
            RuntimeV3Vertex(0, 0, 0, 0, 0),
            RuntimeV3Vertex(1, 0, 0, 1, 0),
            RuntimeV3Vertex(0, 1, 0, 0, 1),
            RuntimeV3Vertex(1, 1, 0, 1, 1),
        ),
        triangles=((0, 1, 2), (1, 3, 2)),
    )
    c = RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        meshes=(body,),
        appearance_patches=(
            RuntimeV3AppearancePatch("p0", "body", (0,), AppearanceProvenance.DIRECT_SOURCE, 0, "V0"),
            RuntimeV3AppearancePatch("p1", "body", (1,), AppearanceProvenance.OTHER_VIEW_SOURCE, 3, "V0_baked_cross_view"),
        ),
    )
    assert len(validate_playback_runtime_v3_contract(c)) == 64


def test_appearance_face_partition_must_be_exact():
    body = RuntimeV3Mesh(
        mesh_id="body",
        slot_id="body_slot",
        attachment_id="body_attachment",
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        vertices=(
            RuntimeV3Vertex(0, 0, 0, 0, 0),
            RuntimeV3Vertex(1, 0, 0, 1, 0),
            RuntimeV3Vertex(0, 1, 0, 0, 1),
            RuntimeV3Vertex(1, 1, 0, 1, 1),
        ),
        triangles=((0, 1, 2), (1, 3, 2)),
    )
    c = RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        meshes=(body,),
        appearance_patches=(
            RuntimeV3AppearancePatch("p0", "body", (0,), AppearanceProvenance.DIRECT_SOURCE, 0, "V0"),
        ),
    )
    with pytest.raises(QualificationError, match="RUNTIME_V3_APPEARANCE_FACE_COVERAGE_INCOMPLETE"):
        validate_playback_runtime_v3_contract(c)


def test_slot_draw_order_is_exact_permutation_and_clip_is_interval_semantics():
    body = _mesh(mesh_id="body", slot_id="body_slot", attachment_id="body_attachment")
    clip = _mesh(
        mesh_id="clip_mesh",
        slot_id="clip_slot",
        attachment_id="clip_attachment",
        kind=AttachmentKind.CLIPPING,
    )
    c = RuntimeV3PlaybackContract(
        slots=(
            RuntimeV3Slot("clip_slot", "root", 0, "clip_attachment"),
            RuntimeV3Slot("body_slot", "root", 1, "body_attachment"),
        ),
        meshes=(clip, body),
        appearance_patches=(
            RuntimeV3AppearancePatch("clip_face", "clip_mesh", (0,), AppearanceProvenance.UNSEEN, None, None),
            RuntimeV3AppearancePatch("body_face", "body", (0,), AppearanceProvenance.DIRECT_SOURCE, 0, "V0"),
        ),
    )
    validate_playback_runtime_v3_contract(c)
    frame = RuntimeV3FrameComposition(
        view_id="V0",
        draw_order_slot_ids=("clip_slot", "body_slot"),
        active_attachment_by_slot={
            "clip_slot": "clip_attachment",
            "body_slot": "body_attachment",
        },
        clip_intervals=(RuntimeV3ClipInterval("clip_attachment", "clip_slot", "body_slot"),),
    )
    validate_frame_composition_v3(c, frame)

    bad = RuntimeV3FrameComposition(
        view_id="V0",
        draw_order_slot_ids=("body_slot", "clip_slot"),
        active_attachment_by_slot=frame.active_attachment_by_slot,
        clip_intervals=frame.clip_intervals,
    )
    with pytest.raises(QualificationError, match="RUNTIME_V3_CLIP_INTERVAL_REVERSED_IN_DRAW_ORDER"):
        validate_frame_composition_v3(c, bad)
