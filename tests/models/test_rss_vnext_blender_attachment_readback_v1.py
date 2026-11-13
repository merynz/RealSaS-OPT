from __future__ import annotations

"""No-Blender readback validation of original teacher + sampled parent/root.

All fixtures are synthetic. Results cannot qualify training, contact, or a
NO_FACE negative without an external source-native authored inventory court.
"""
from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
import pytest

from tools.truth_foundry.native_blender_attachment_inventory_v1 import (
    compile_blender_native_attachment_inventory_candidate_v1,
)
from tools.truth_foundry.audit_blender_native_attachment_receipt_v1 import (
    audit_native_blender_attachment_receipt_v1,
)

def _owner(i,name,bone=""):
    return {
        "owner_index":i,"name":name,
        "parent_type":"BONE" if bone else "OBJECT",
        "parent_name":"Rig" if bone else None,
        "parent_object_type":"ARMATURE" if bone else None,
        "parent_bone":bone,
        "constraints":[],"modifiers":[],
    }


def _mat(dx=0.,angle=0.):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([
        [c,-s,0.,dx],[s,c,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],
    ])


def _fixture():
    authors=[_owner(0,"Sword","Grip"),_owner(1,"Body")]
    faces_owner=np.asarray([0,0,1,1,1],dtype=np.int32)
    registry=compile_blender_native_attachment_inventory_candidate_v1(
        owner_rows=authors,source_face_owner=faces_owner,
        source_asset_sha256="a"*64,
        native_teacher_receipt_sha256="b"*64,
        native_teacher_npz_sha256="c"*64,
        observation_admission_sha256="d"*64,
    )
    rows=[
        {"action":"Swing","armature":"Rig","frame":1},
        {"action":"Swing","armature":"Rig","frame":4},
        {"action":"Other","armature":"Other","frame":1},
    ]
    rest=np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [2,0,0],[3,0,0],[2,1,0],
    ],dtype=np.float64)
    faces=np.array([
        [0,1,2],[0,2,1],[3,4,5],[3,5,4],[4,3,5],
    ],dtype=np.int64)
    teacher={
        "schema":"RealSaS.IRISVNextAuthoredTeacherExport.v1",
        "subject_id":"test-subject",
        "source":{"sha256":"a"*64},
        "output":{"sha256":"c"*64},
        "observation_admission_sha256":"d"*64,
        "owners":[
            {"source_import_name":"Sword","anonymous_owner_index":0},
            {"source_import_name":"Body","anonymous_owner_index":1},
        ],
        "anonymous_owner_count":2,"face_count":5,
        "sampled_frames":rows,
        "import":{"operation":"GLTF_IMPORT_WITH_NATIVE_ANIMATION"},
    }
    original={
        "faces":faces,
        "face_owner":faces_owner,
        "rest_vertices_source_world":rest,
        "posed_vertices_source_world":np.stack((rest,rest,rest)),
    }
    frames={
        "parent_rest_world":np.full((2,4,4),np.nan,dtype=np.float64),
        "root_rest_world":np.full((2,4,4),np.nan,dtype=np.float64),
        "parent_sampled_world":np.full((2,3,4,4),np.nan,dtype=np.float64),
        "root_sampled_world":np.full((2,3,4,4),np.nan,dtype=np.float64),
        "sample_available":np.array(
            [[True,True,False],[False,False,False]],dtype=np.bool_,
        ),
    }
    frames["parent_rest_world"][0]=_mat()
    frames["root_rest_world"][0]=_mat()
    for frame in (0,1):
        frames["parent_sampled_world"][0,frame]=_mat(.1*(frame+1),.2*(frame+1))
        frames["root_sampled_world"][0,frame]=_mat()
    rig_graph=[
        {
            "armature_name":"Rig","object_parent":None,
            "object_constraints":[],
            "bones":[{
                "name":"Grip","parent":None,
                "native_rest_matrix_local":_mat().tolist(),
                "constraints":[],
            }],
            "native_graph_authority":"DIAGNOSTIC_ONLY_NO_HOLDER_FACE_OR_SOCKET_ID",
        },
        {
            "armature_name":"Other","object_parent":None,
            "object_constraints":[],"bones":[],
            "native_graph_authority":"DIAGNOSTIC_ONLY_NO_HOLDER_FACE_OR_SOCKET_ID",
        },
    ]
    graph_sha=sha256(json.dumps(
        rig_graph,sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()
    receipt={
        "schema":"RealSaS.TruthFoundry.BlenderNativeAttachmentObservation.v1",
        "subject_id":"test-subject",
        "source_asset_sha256":"a"*64,
        "teacher_receipt_sha256":"b"*64,
        "teacher_npz_sha256":"c"*64,
        "observation_admission_sha256":"d"*64,
        "importer_policy":teacher["import"],
        "original_teacher_frame_rows":rows,
        "source_owner_inventory":authors,
        "native_armature_bone_constraint_graph":rig_graph,
        "native_armature_bone_constraint_graph_sha256":graph_sha,
        "source_link_candidates":json.loads(json.dumps(asdict(registry))),
        "parent_root_npz_sha256":"e"*64,
        "parent_root_npz_filename":"native_attachment_parent_root_frames.npz",
        "source_coordinate_domain":"SOURCE_WORLD_NOT_REALSAS_CANONICAL",
        "attachment_face_authority":False,
        "exhaustive_no_face_authority":False,
        "trainer_admission":False,
        "product_revision_authority":False,
    }
    kwargs=dict(
        receipt=receipt,teacher_receipt=teacher,
        teacher_npz=original,parent_root_npz=frames,
        source_asset_sha256="a"*64,
        teacher_receipt_sha256="b"*64,
        teacher_npz_sha256="c"*64,
        observation_admission_sha256="d"*64,
        parent_root_npz_sha256="e"*64,
    )
    return kwargs


def test_readback_recomputes_native_face_owner_and_samples_against_original_teacher():
    q=_fixture()
    out=audit_native_blender_attachment_receipt_v1(**q)
    assert out["schema"]=="RealSaS.TruthFoundry.BlenderNativeAttachmentReadback.v1"
    assert out["bone_link_candidate_count"]==1
    assert out["native_source_armature_count"]==2
    assert out["native_source_bone_count"]==1
    assert out["parent_root_available_samples"]==2
    assert out["original_teacher_frame_count"]==3
    assert out["non_se3_source_frames"]==0
    assert not out["source_attachment_training_admitted"]
    assert not out["source_attachment_contact_qualified"]
    assert not out["exhaustive_no_face_verified"]
    assert not out["heldout_intervention_qualified"]


def test_forged_authority_flags_or_teacher_receipt_hash_are_rejected():
    q=_fixture()
    for field in (
        "trainer_admission","attachment_face_authority",
        "exhaustive_no_face_authority","product_revision_authority",
    ):
        bad={**q,"receipt":{**q["receipt"],field:True}}
        with pytest.raises(ValueError,match="PARENT_OR_AUTHORITY_DRIFT"):
            audit_native_blender_attachment_receipt_v1(**bad)
    bad={**q,"teacher_receipt_sha256":"f"*64}
    with pytest.raises(ValueError,match="PARENT_OR_AUTHORITY_DRIFT"):
        audit_native_blender_attachment_receipt_v1(**bad)


def test_authoring_inventory_spoof_cannot_rebind_original_face_owner():
    q=_fixture()
    changed=json.loads(json.dumps(q["receipt"]))
    changed["source_owner_inventory"][0]["parent_bone"]="OtherBone"
    with pytest.raises(ValueError,match="AUTHORING_LEDGER_OR_ORIGINAL_TEACHER_OWNERS_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"receipt":changed},
        )
    changed=q["teacher_npz"].copy()
    changed["face_owner"]=np.array([0,1,1,1,1],dtype=np.int32)
    with pytest.raises(ValueError,match="AUTHORING_LEDGER_OR_ORIGINAL_TEACHER_OWNERS_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"teacher_npz":changed},
        )


def test_changed_action_frame_identity_cannot_replay_as_same_attachment_teacher():
    q=_fixture()
    changed=json.loads(json.dumps(q["receipt"]))
    changed["original_teacher_frame_rows"][1]["frame"]=2
    with pytest.raises(ValueError,match="PARENT_OR_AUTHORITY_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"receipt":changed},
        )


def test_fake_pose_sample_on_different_armature_must_be_missing():
    q=_fixture()
    changed={k:v.copy() for k,v in q["parent_root_npz"].items()}
    changed["sample_available"][0,2]=True
    changed["parent_sampled_world"][0,2]=_mat()
    changed["root_sampled_world"][0,2]=_mat()
    with pytest.raises(ValueError,match="SOURCE_PARENT_ACTION_FRAME_MASK_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"parent_root_npz":changed},
        )


def test_missing_frame_cannot_hide_data_in_nan_slot():
    q=_fixture()
    changed={k:v.copy() for k,v in q["parent_root_npz"].items()}
    changed["parent_sampled_world"][0,2]=_mat()
    with pytest.raises(ValueError,match="MISSING_ACTION_FRAME_WAS_SYNTHESIZED"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"parent_root_npz":changed},
        )


def test_nonhomogeneous_or_singular_parent_frame_cannot_be_measurement_truth():
    q=_fixture()
    changed={k:v.copy() for k,v in q["parent_root_npz"].items()}
    changed["parent_sampled_world"][0,0,3,0]=.5
    with pytest.raises(ValueError,match="AVAILABLE_FRAME_NOT_HOMOGENEOUS"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"parent_root_npz":changed},
        )
    changed={k:v.copy() for k,v in q["parent_root_npz"].items()}
    changed["parent_sampled_world"][0,0,:3,:3]=np.zeros((3,3))
    with pytest.raises(ValueError,match="AVAILABLE_FRAME_SINGULAR"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"parent_root_npz":changed},
        )


def test_valid_nonuniform_source_scale_is_diagnostic_not_se3_certification():
    q=_fixture()
    changed={k:v.copy() for k,v in q["parent_root_npz"].items()}
    changed["parent_sampled_world"][0,0,0,0]=2.
    out=audit_native_blender_attachment_receipt_v1(
        **{**q,"parent_root_npz":changed},
    )
    assert out["non_se3_source_frames"]==1
    assert not out["heldout_intervention_qualified"]


def test_rig_graph_hash_parent_and_source_action_identity_are_immutable():
    q=_fixture()
    bad=json.loads(json.dumps(q["receipt"]))
    bad["native_armature_bone_constraint_graph"][0]["bones"][0]["parent"]="WrongBone"
    with pytest.raises(ValueError,match="NATIVE_SOURCE_RIG_GRAPH_HASH_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"receipt":bad},
        )
    graph=bad["native_armature_bone_constraint_graph"]
    bad["native_armature_bone_constraint_graph_sha256"]=sha256(json.dumps(
        graph,sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()
    with pytest.raises(ValueError,match="SOURCE_RIG_PARENT_OR_CONSTRAINT_DRIFT"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"receipt":bad},
        )
    bad=json.loads(json.dumps(q["receipt"]))
    bad["native_armature_bone_constraint_graph"][1]["armature_name"]="Unrelated"
    bad["native_armature_bone_constraint_graph_sha256"]=sha256(json.dumps(
        bad["native_armature_bone_constraint_graph"],
        sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()
    with pytest.raises(ValueError,match="SOURCE_RIG_ACTION_OWNER_UNRESOLVED"):
        audit_native_blender_attachment_receipt_v1(
            **{**q,"receipt":bad},
        )
