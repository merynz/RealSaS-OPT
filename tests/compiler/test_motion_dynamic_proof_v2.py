import math
import numpy as np

from compiler.realsas_compiler_core.motion_compile_v2 import (
    CanonicalJointTrack3DIR,
    MotionKeyframe3DIR,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3


class Joint:
    def __init__(self,jid,p,parent=None):
        self.canonical_joint_id=jid
        self.position=p
        self.parent_canonical_id=parent


class Skeleton:
    def __init__(self):
        self.joints=(
            Joint("root",(0.0,0.0,0.0),None),
            Joint("child",(0.0,0.0,1.0),"root"),
        )
        self.root_id="root"
        self.skeleton_lineage_hash="sk"


def _cameras():
    rows=[]
    for i in range(8):
        a=math.radians(45.0*i)
        forward=(-math.sin(a),-math.cos(a),0.0)
        right=(-math.cos(a),math.sin(a),0.0)
        rows.append(
            CameraProjectionV3(
                f"V{i}",i,(0.0,0.0,0.0),right,(0.0,0.0,1.0),
                forward,1.0,64
            )
        )
    return tuple(rows)


def test_quaternion_fk_moves_child_out_of_legacy_xy_rotation_plane():
    # 90 degrees about the derived root local X axis.
    s=math.sqrt(0.5)
    track=CanonicalJointTrack3DIR(
        "root","source_root",("LOCAL_ROTATION_QUAT_XYZW",),
        (
            MotionKeyframe3DIR(0.0,(0.0,0.0,0.0,1.0)),
            MotionKeyframe3DIR(1.0,(s,0.0,0.0,s)),
        ),
    )
    skin,pos,frame_hash=_joint_pose_v2(
        skeleton=Skeleton(),
        tracks={"root":track},
        time_seconds=1.0,
        cameras=_cameras(),
    )
    child=np.asarray(pos["child"])
    assert frame_hash
    # Bone length preserved and the child is no longer on the original +Z line.
    assert np.isclose(np.linalg.norm(child-np.asarray(pos["root"])),1.0,atol=1e-9)
    assert abs(child[1])>0.9
    assert abs(child[2])<1e-8
