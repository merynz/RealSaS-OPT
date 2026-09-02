from __future__ import annotations

from .hashing import content_sha256
from .types import QualificationError
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR


def _select_preset_joint(skeleton):
    joints=tuple(sorted(skeleton.joints,key=lambda j:j.canonical_joint_id))
    if not joints: raise QualificationError("MOTION_REQUIRES_QUALIFIED_JOINT")
    children=[j for j in joints if j.parent_canonical_id is not None]
    return (children or list(joints))[0]


def build_deterministic_preset_motion(mechanical, *, clip_id:str="preset_idle_v1", duration_sec:float=1.0, amplitude_deg:float=4.0):
    """Build a topology-only puppet-local preset. No authored joint names/3D rotations."""
    if duration_sec <= 0.0: raise ValueError("duration_sec must be positive")
    if not (0.0 < abs(float(amplitude_deg)) <= 45.0): raise ValueError("bounded nonzero amplitude required")
    joint=_select_preset_joint(mechanical.skeleton)
    spec={"schema":"RealSaS.DeterministicPresetMotion.v1","clip_id":clip_id,"duration_sec":float(duration_sec),"joint_id":joint.canonical_joint_id,"joint_parent":joint.parent_canonical_id,"amplitude_deg":float(amplitude_deg),"transform_space":"PUPPET_LOCAL_2D_2P5D","selection_policy":"SORTED_FIRST_NONROOT_ELSE_FIRST_ROOT"}
    clip_hash=content_sha256(spec)
    clip=MotionClipIR(str(clip_id),"PRESET",clip_hash,("PRESET_MOTION",),source_ref="RealSaS.MotionCompiler.DeterministicPreset.v1",duration_sec=float(duration_sec),loop=True,metadata={"semantic_joint_name_used":False,"quaternion_used":False,"vec3_rigid_motion_used":False,"spec_hash":clip_hash})
    keys=(
        JointTransformKeyIR(0.0,(0.0,0.0),0.0,(1.0,1.0),0.0),
        JointTransformKeyIR(float(duration_sec)*0.25,(0.0,0.0),float(amplitude_deg),(1.0,1.0),0.0),
        JointTransformKeyIR(float(duration_sec)*0.75,(0.0,0.0),-float(amplitude_deg),(1.0,1.0),0.0),
        JointTransformKeyIR(float(duration_sec),(0.0,0.0),0.0,(1.0,1.0),0.0),
    )
    track=build_joint_track("TRACK:"+clip_hash[:16],clip.clip_id,joint.canonical_joint_id,keys,metadata={"joint_selection_policy":"TOPOLOGY_ONLY","authored_joint_name_dependency":False})
    state=build_motion_state((clip,),(track,),metadata={"producer":"RealSaS.MotionCompiler.DeterministicPreset.v1","authored_joint_names_used":False,"full_3d_motion_authority":False})
    validate_motion_against_mechanical(state,mechanical)
    return state
