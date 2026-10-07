from __future__ import annotations

"""Carrier-native deformation conditioning for the AXIS/MIRA line.

Stage34 derives a subject-free pre-bind frame witness from skeleton geometry and
camera basis only. Coincident edges may provisionally inherit the parent frame;
that is not a mechanical acceptance decision. Stage35 must rederive and qualify
the same frames against final carrier-native W_M before dynamic proof.
"""

from dataclasses import replace

from .hashing import content_sha256
from .joint_frames_v2 import derive_joint_frames_pre_bind_v2, frame_set_hash_pre_bind_v2
from .mesh.deformation_stress_v1 import expected_g3_probe_plan_hash
from .motion_3d_adapter_v1 import parse_axis_contract_v1
from .product_authority_v1 import (
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    deformation_envelope_lineage_hash,
    validate_deformation_capability_envelope,
)
from .types import QualificationError
from .deformation_envelope_derivation_v1 import GENERIC_DEFORMATION_POLICY_V1


def generic_deformation_policy_hash_v2() -> str:
    return content_sha256(
        {
            **dict(GENERIC_DEFORMATION_POLICY_V1),
            "schema": "RealSaS.GenericDeformationEnvelopePolicy.v2",
            "joint_frame_semantics": "PRE_BIND_PROVISIONAL__POST_BIND_STAGE35_REQUIRED",
        }
    )


def derive_axis_contract_v2(*, skeleton, camera_set):
    joints = tuple(skeleton.joints)
    if not joints:
        raise QualificationError("AUTO_ENVELOPE_V2_SKELETON_EMPTY")
    known = {str(j.canonical_joint_id) for j in joints}
    if len(known) != len(joints) or str(skeleton.root_id) not in known:
        raise QualificationError("AUTO_ENVELOPE_V2_SKELETON_INVALID")
    children = {jid: 0 for jid in known}
    for joint in joints:
        parent = joint.parent_canonical_id
        if parent is not None:
            parent = str(parent)
            if parent not in known:
                raise QualificationError("AUTO_ENVELOPE_V2_PARENT_UNKNOWN")
            children[parent] += 1

    frames, frame_report = derive_joint_frames_pre_bind_v2(
        skeleton,
        cameras=camera_set.cameras,
    )
    rows = []
    for joint in sorted(joints, key=lambda row: str(row.canonical_joint_id)):
        jid = str(joint.canonical_joint_id)
        role = (
            "TOPOLOGY_ROOT"
            if jid == str(skeleton.root_id)
            else ("TOPOLOGY_LEAF" if children[jid] == 0 else "TOPOLOGY_INTERNAL")
        )
        R = frames[jid].rotation_matrix
        rows.append(
            {
                "canonical_joint_id": jid,
                "axis_xyz": [float(R[0][0]), float(R[1][0]), float(R[2][0])],
                "legacy_scalar_to_semantic_sign": 1.0,
                "role": role,
                "derived_joint_frame_hash": frames[jid].frame_hash,
            }
        )
    value = {
        "schema": "RealSaS.DerivedAxisContract.v1",
        "status": "PASS_COMPATIBILITY_WITNESS_ONLY",
        "skeleton_binding_hash": skeleton.skeleton_lineage_hash,
        "camera_set_binding_hash": camera_set.camera_set_hash,
        "derived_joint_frame_set_hash": frame_set_hash_pre_bind_v2(frames),
        "joint_frame_qualification": frame_report,
        "policy_hash": generic_deformation_policy_hash_v2(),
        "joint_axes": rows,
        "subject_specific_authoring_used": False,
        "categorical_recognition_used": False,
        "actual_motion_capability_claimed": False,
        "post_bind_qualification_required": True,
    }
    _, axis_hash = parse_axis_contract_v1(value)
    value["axis_contract_hash"] = axis_hash
    return value


def derive_deformation_envelope_v2(*, skeleton, camera_set):
    axis = derive_axis_contract_v2(
        skeleton=skeleton,
        camera_set=camera_set,
    )
    _, axis_hash = parse_axis_contract_v1(axis)
    limit = float(GENERIC_DEFORMATION_POLICY_V1["rotation_limit_deg"])
    ranges = tuple(
        JointCapabilityRangeIR(
            j.canonical_joint_id,
            -limit,
            limit,
            0.0,
            1.0,
            1.0,
            metadata={
                "derivation": "GENERIC_SUBJECT_FREE_PRE_BIND_V2",
                "subject_authored": False,
            },
        )
        for j in sorted(skeleton.joints, key=lambda row: row.canonical_joint_id)
    )
    camera_hashes = tuple(camera_set.camera_binding_hashes)
    probe = expected_g3_probe_plan_hash(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        axis_contract_hash=axis_hash,
        joint_ranges=ranges,
        camera_binding_hashes=camera_hashes,
        allowed_attachment_state_hashes=(),
    )
    env = DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        joint_ranges=ranges,
        camera_binding_hashes=camera_hashes,
        allowed_attachment_state_hashes=(),
        axis_contract_hash=axis_hash,
        probe_plan_hash=probe,
        envelope_lineage_hash="",
        metadata={
            "producer": "RealSaS.GenericDeformationConditioningLineage.v3",
            "policy": dict(GENERIC_DEFORMATION_POLICY_V1),
            "policy_hash": generic_deformation_policy_hash_v2(),
            "camera_set_hash": camera_set.camera_set_hash,
            "joint_frame_semantics": "PRE_BIND_PROVISIONAL__POST_BIND_STAGE35_REQUIRED",
            "post_bind_qualification_required": True,
            "per_character_joint_range_authoring": False,
            "per_character_axis_authoring": False,
            "translation_scale_probe_support": "IDENTITY_ONLY_V1",
            "actual_motion_capability_claimed": False,
            "actual_motion_capability_authority": "STAGE41_EXACT_MOTION_PROOF",
            "g3_role": "LOCAL_3D_NUMERICAL_CONDITIONING_ONLY",
            "skin_topology_compatibility_stress_angle_deg": 120.0,
            "skin_topology_compatibility_stress_semantics": (
                "SUBJECT_FREE_MECHANICAL_STRESS__NOT_MOTION_CAPABILITY"
            ),
        },
    )
    env = replace(
        env, envelope_lineage_hash=deformation_envelope_lineage_hash(env)
    )
    validate_deformation_capability_envelope(
        env, known_joint_ids={j.canonical_joint_id for j in skeleton.joints}
    )
    return axis, env


__all__ = [
    "derive_axis_contract_v2",
    "derive_deformation_envelope_v2",
    "generic_deformation_policy_hash_v2",
]
