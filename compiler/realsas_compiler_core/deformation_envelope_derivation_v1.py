from __future__ import annotations
from dataclasses import replace

from .hashing import content_sha256
from .mesh.deformation_stress_v1 import expected_g3_probe_plan_hash
from .motion_3d_adapter_v1 import parse_axis_contract_v1
from .joint_frames_v1 import derive_joint_frames_from_skeleton, frame_set_hash
from .product_authority_v1 import (
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    deformation_envelope_lineage_hash,
    validate_deformation_capability_envelope,
)
from .types import QualificationError

GENERIC_DEFORMATION_POLICY_V1={
    "schema":"RealSaS.GenericDeformationEnvelopePolicy.v1",
    "status":"FROZEN_PRE_KNIGHT_SUBJECT_FREE",
    "rotation_limit_deg":10.0,
    "translation_radius":0.0,
    "min_scale":1.0,
    "max_scale":1.0,
    "rotation_semantics":"LEGACY_SCALAR_COMPATIBILITY_WITNESS_ON_DERIVED_LOCAL_X",
    "actual_motion_capability_authority":"STAGE35_EXACT_QUATERNION_CLIP_EXECUTION",
}

def generic_deformation_policy_hash_v1():
    return content_sha256(GENERIC_DEFORMATION_POLICY_V1)

def derive_axis_contract_v1(skeleton,camera_set):
    joints=tuple(skeleton.joints)
    if not joints:
        raise QualificationError("AUTO_ENVELOPE_SKELETON_EMPTY")
    known={j.canonical_joint_id for j in joints}
    if len(known)!=len(joints) or skeleton.root_id not in known:
        raise QualificationError("AUTO_ENVELOPE_SKELETON_INVALID")
    children={jid:0 for jid in known}
    for joint in joints:
        if joint.parent_canonical_id is not None:
            if joint.parent_canonical_id not in known:
                raise QualificationError("AUTO_ENVELOPE_PARENT_UNKNOWN")
            children[joint.parent_canonical_id]+=1
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=camera_set.cameras)
    rows=[]
    for joint in sorted(joints,key=lambda x:x.canonical_joint_id):
        jid=joint.canonical_joint_id
        role="TOPOLOGY_ROOT" if jid==skeleton.root_id else ("TOPOLOGY_LEAF" if children[jid]==0 else "TOPOLOGY_INTERNAL")
        R=frames[jid].rotation_matrix
        local_x=[float(R[0][0]),float(R[1][0]),float(R[2][0])]
        rows.append({
            "canonical_joint_id":jid,
            "axis_xyz":local_x,
            "legacy_scalar_to_semantic_sign":1.0,
            "role":role,
            "derived_joint_frame_hash":frames[jid].frame_hash,
        })
    value={
        "schema":"RealSaS.DerivedAxisContract.v1",
        "status":"PASS_COMPATIBILITY_WITNESS_ONLY",
        "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
        "camera_set_binding_hash":camera_set.camera_set_hash,
        "derived_joint_frame_set_hash":frame_set_hash(frames),
        "policy_hash":generic_deformation_policy_hash_v1(),
        "joint_axes":rows,
        "subject_specific_authoring_used":False,
        "categorical_recognition_used":False,
        "actual_motion_capability_claimed":False,
    }
    _,axis_hash=parse_axis_contract_v1(value)
    value["axis_contract_hash"]=axis_hash
    return value
def derive_deformation_envelope_v1(*,skeleton,camera_set):
    axis=derive_axis_contract_v1(skeleton,camera_set)
    _,axis_hash=parse_axis_contract_v1(axis)
    limit=float(GENERIC_DEFORMATION_POLICY_V1["rotation_limit_deg"])
    ranges=tuple(
        JointCapabilityRangeIR(
            j.canonical_joint_id,-limit,limit,0.0,1.0,1.0,
            metadata={"derivation":"GENERIC_SUBJECT_FREE_CANONICAL_XY_V1","subject_authored":False}
        )
        for j in sorted(skeleton.joints,key=lambda x:x.canonical_joint_id)
    )
    camera_hashes=tuple(camera_set.camera_binding_hashes)
    probe=expected_g3_probe_plan_hash(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        axis_contract_hash=axis_hash,
        joint_ranges=ranges,
        camera_binding_hashes=camera_hashes,
        allowed_attachment_state_hashes=(),
    )
    env=DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        joint_ranges=ranges,
        camera_binding_hashes=camera_hashes,
        allowed_attachment_state_hashes=(),
        axis_contract_hash=axis_hash,
        probe_plan_hash=probe,
        envelope_lineage_hash="",
        metadata={
            "producer":"RealSaS.GenericDeformationConditioningLineage.v2",
            "policy":dict(GENERIC_DEFORMATION_POLICY_V1),
            "policy_hash":generic_deformation_policy_hash_v1(),
            "camera_set_hash":camera_set.camera_set_hash,
            "per_character_joint_range_authoring":False,
            "per_character_axis_authoring":False,
            "translation_scale_probe_support":"IDENTITY_ONLY_V1",
            "actual_motion_capability_claimed":False,
            "actual_motion_capability_authority":"STAGE35_EXACT_QUATERNION_CLIP_EXECUTION",
            "g3_role":"LOCAL_3D_NUMERICAL_CONDITIONING_ONLY",
        },
    )
    env=replace(env,envelope_lineage_hash=deformation_envelope_lineage_hash(env))
    validate_deformation_capability_envelope(env,known_joint_ids={j.canonical_joint_id for j in skeleton.joints})
    return axis,env
