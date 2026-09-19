import numpy as np

from compiler.realsas_compiler_core.joint_frames_v1 import (
    derive_joint_frames_from_rows,
    frame_set_hash,
)
from compiler.realsas_compiler_core.motion_compile_v2 import automatic_retarget_map_v2


class J:
    def __init__(self,jid,p,parent=None):
        self.canonical_joint_id=jid
        self.position=p
        self.parent_canonical_id=parent


class S:
    def __init__(self):
        self.joints=(
            J("root",(0,0,0),None),
            J("hipL",(-.3,0,-.4),"root"),
            J("footL",(-.3,0,-1.0),"hipL"),
            J("hipR",(.3,0,-.4),"root"),
            J("footR",(.3,0,-1.0),"hipR"),
        )
        self.root_id="root"
        self.skeleton_lineage_hash="sk"


def test_joint_frames_are_deterministic_and_right_handed():
    rows=(
        {"id":"root","parent":None,"p":(0,0,0)},
        {"id":"a","parent":"root","p":(0,0,1)},
        {"id":"b","parent":"a","p":(.5,0,1.5)},
    )
    a=derive_joint_frames_from_rows(
        rows,
        joint_id_key="id",
        parent_id_key="parent",
        position_key="p",
    )
    b=derive_joint_frames_from_rows(
        rows,
        joint_id_key="id",
        parent_id_key="parent",
        position_key="p",
    )
    assert frame_set_hash(a)==frame_set_hash(b)
    for frame in a.values():
        R=np.asarray(frame.rotation_matrix)
        assert np.allclose(R.T@R,np.eye(3),atol=1e-9)
        assert np.linalg.det(R)>0.999999


def test_retarget_allows_extra_source_joints_but_preserves_target_tree():
    payload={
        "coordinate_frame":"REALSAS_OBJECT_FRAME_V1",
        "source_skeleton":[
            {"source_joint_id":"hips","parent_source_joint_id":None,"rest_position":[0,0,0]},
            {"source_joint_id":"thighL","parent_source_joint_id":"hips","rest_position":[-.3,0,-.4]},
            {"source_joint_id":"twistL","parent_source_joint_id":"thighL","rest_position":[-.3,0,-.7]},
            {"source_joint_id":"footL","parent_source_joint_id":"twistL","rest_position":[-.3,0,-1]},
            {"source_joint_id":"thighR","parent_source_joint_id":"hips","rest_position":[.3,0,-.4]},
            {"source_joint_id":"twistR","parent_source_joint_id":"thighR","rest_position":[.3,0,-.7]},
            {"source_joint_id":"footR","parent_source_joint_id":"twistR","rest_position":[.3,0,-1]},
        ],
    }
    mapping,report=automatic_retarget_map_v2(payload,S())
    assert len(mapping)==5
    assert mapping["hips"]=="root"
    assert set(mapping.values())=={"root","hipL","footL","hipR","footR"}
    assert report["unused_source_joint_count"]==2
