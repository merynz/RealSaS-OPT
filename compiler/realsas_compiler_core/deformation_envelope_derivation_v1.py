from __future__ import annotations
from dataclasses import replace

from .hashing import content_sha256
from .mesh.deformation_stress_v1 import expected_g3_probe_plan_hash
from .motion_3d_adapter_v1 import parse_axis_contract_v1
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
    "rotation_limit_deg":45.0,
    "translation_radius":0.0,
    "min_scale":1.0,
    "max_scale":1.0,
    "canonical_rotation_axis_xyz":(0.0,0.0,1.0),
    "rotation_semantics":"CANONICAL_XY_AROUND_POSITIVE_Z",
}

def generic_deformation_policy_hash_v1():
    return content_sha256(GENERIC_DEFORMATION_POLICY_V1)

def derive_axis_contract_v1(skeleton):
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
    rows=[]
    for joint in sorted(joints,key=lambda x:x.canonical_joint_id):
        jid=joint.canonical_joint_id
        role="TOPOLOGY_ROOT" if jid==skeleton.root_id else ("TOPOLOGY_LEAF" if children[jid]==0 else "TOPOLOGY_INTERNAL")
        rows.append({
            "canonical_joint_id":jid,
            "axis_xyz":[0.0,0.0,1.0],
            "legacy_scalar_to_semantic_sign":1.0,
            "role":role,
        })
    value={
        "schema":"RealSaS.DerivedAxisContract.v1",
        "status":"PASS",
        "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
        "policy_hash":generic_deformation_policy_hash_v1(),
        "joint_axes":rows,
        "subject_specific_authoring_used":False,
        "categorical_recognition_used":False,
    }
    _,axis_hash=parse_axis_contract_v1(value)
    value["axis_contract_hash"]=axis_hash
    return value

def derive_deformation_envelope_v1(*,skeleton,camera_set):
    axis=derive_axis_contract_v1(skeleton)
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
            "producer":"RealSaS.GenericDeformationEnvelopeDerivation.v1",
            "policy":dict(GENERIC_DEFORMATION_POLICY_V1),
            "policy_hash":generic_deformation_policy_hash_v1(),
            "camera_set_hash":camera_set.camera_set_hash,
            "per_character_joint_range_authoring":False,
            "per_character_axis_authoring":False,
            "translation_scale_probe_support":"IDENTITY_ONLY_V1",
        },
    )
    env=replace(env,envelope_lineage_hash=deformation_envelope_lineage_hash(env))
    validate_deformation_capability_envelope(env,known_joint_ids={j.canonical_joint_id for j in skeleton.joints})
    return axis,env
