import copy
import numpy as np
import pytest
from compiler.realsas_compiler_core.presentation_attachment_v1 import (
    qualify_target_attachments, qualify_body_motion_preset, visual_face_attachment_owners,
)
from compiler.realsas_compiler_core.types import QualificationError


def fixture():
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]], dtype=float)
    f = np.array([[0,1,2],[3,4,5]])
    contract = {"schema":"RealSaS.TargetPresentationAttachments.v1", "carrier_basis_sha256":"M",
        "skeleton_sha256":"G", "attachments":[{"attachment_id":"SWORD", "component_indices":[1],
        "target_slot_canonical_joint_id":"HAND"}]}
    return v, f, contract


def qualify(c):
    v,f,_ = fixture()
    return qualify_target_attachments(c, vertices=v, faces=f, carrier_basis_sha256="M",
                                      skeleton_sha256="G", canonical_to_raw={"HAND":13})


def test_unarmed_body_preset_keeps_target_attachment_and_rest_placement():
    v,f,c = fixture(); owners, target = qualify(c)
    np.testing.assert_array_equal(owners,[0,0,0,1,1,1])
    preset = qualify_body_motion_preset({"schema":"RealSaS.BodyMotionPreset.v1",
        "semantic_scope":"BODY_KINEMATIC_DELTAS_ONLY", "clip_sha256":["IDLE","RUN","SLASH"],
        "source_attachment_geometry_required":False,"target_attachment_geometry_required":False},
        clip_hashes=["IDLE","RUN","SLASH"],target_attachments=target)
    assert preset["optional_target_attachments"] == ["SWORD"]
    assert preset["frame0_is_rest_pose"] is False
    np.testing.assert_array_equal(visual_face_attachment_owners({"anchor_vertex":np.array([0,3]),
        "anchor_mechanical_vertices":f, "domain_id":np.array([0,0,0,1,1,1])},f,owners),[0,1])


def test_overlap_or_changed_canonical_authority_rejected():
    _,_,c = fixture(); c["attachments"].append({**c["attachments"][0],"attachment_id":"SHIELD"})
    with pytest.raises(QualificationError, match="OWNERSHIP_CONFLICT"): qualify(c)
    c=fixture()[2]; c["carrier_basis_sha256"]="OTHER"
    with pytest.raises(QualificationError,match="AUTHORITY_DRIFT"): qualify(c)


def test_chart_cannot_silently_mix_body_and_prop_ownership():
    _,f,c=fixture(); owners,_=qualify(c)
    with pytest.raises(QualificationError,match="DOMAIN_OWNERSHIP_AMBIGUOUS"):
        visual_face_attachment_owners({"anchor_vertex":np.array([0,3]),
            "anchor_mechanical_vertices":f,"domain_id":np.zeros(6,dtype=int)}, f, owners)
