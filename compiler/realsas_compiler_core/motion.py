from __future__ import annotations

import math

from .hashing import content_sha256
from .types import QualificationError
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR


def _select_preset_joint(mechanical):
    joints = tuple(sorted(mechanical.skeleton.joints, key=lambda j: j.canonical_joint_id))
    if not joints:
        raise QualificationError("MOTION_REQUIRES_QUALIFIED_JOINT")
    if not mechanical.skin.rows:
        raise QualificationError("MOTION_REQUIRES_QUALIFIED_SKIN_SUPPORT")

    mass = {j.canonical_joint_id: 0.0 for j in joints}
    for row in mechanical.skin.rows:
        for jid, value in row.influences:
            if jid not in mass:
                raise QualificationError("MOTION_SKIN_REFERENCES_UNKNOWN_CANONICAL_JOINT")
            w = float(value)
            if not math.isfinite(w) or w < 0.0:
                raise QualificationError("MOTION_INVALID_QUALIFIED_SKIN_WEIGHT")
            mass[jid] += w

    ranked = sorted(joints, key=lambda j: (-mass[j.canonical_joint_id], j.canonical_joint_id))
    joint = ranked[0]
    if mass[joint.canonical_joint_id] <= 0.0:
        raise QualificationError("MOTION_NO_SKIN_SUPPORTED_CANONICAL_JOINT")
    return joint, float(mass[joint.canonical_joint_id])


def build_deterministic_preset_motion(mechanical, *, clip_id: str = "preset_idle_v1", duration_sec: float = 1.0, amplitude_deg: float = 4.0):
    """Build a qualified-skin-supported puppet-local preset without semantic joint names."""
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not (0.0 < abs(float(amplitude_deg)) <= 45.0):
        raise ValueError("bounded nonzero amplitude required")

    joint, selected_skin_mass = _select_preset_joint(mechanical)
    selection_policy = "MAX_QUALIFIED_SKIN_MASS_THEN_CANONICAL_ID"
    spec = {
        "schema": "RealSaS.DeterministicPresetMotion.v2",
        "clip_id": clip_id,
        "duration_sec": float(duration_sec),
        "joint_id": joint.canonical_joint_id,
        "joint_parent": joint.parent_canonical_id,
        "selected_joint_skin_mass": selected_skin_mass,
        "amplitude_deg": float(amplitude_deg),
        "transform_space": "PUPPET_LOCAL_2D_2P5D",
        "selection_policy": selection_policy,
    }
    clip_hash = content_sha256(spec)
    clip = MotionClipIR(
        str(clip_id),
        "PRESET",
        clip_hash,
        ("PRESET_MOTION",),
        source_ref="RealSaS.MotionCompiler.DeterministicPreset.v2",
        duration_sec=float(duration_sec),
        loop=True,
        metadata={
            "semantic_joint_name_used": False,
            "quaternion_used": False,
            "vec3_rigid_motion_used": False,
            "spec_hash": clip_hash,
            "joint_selection_policy": selection_policy,
            "selected_joint_skin_mass": selected_skin_mass,
        },
    )
    keys = (
        JointTransformKeyIR(0.0, (0.0, 0.0), 0.0, (1.0, 1.0), 0.0),
        JointTransformKeyIR(float(duration_sec) * 0.25, (0.0, 0.0), float(amplitude_deg), (1.0, 1.0), 0.0),
        JointTransformKeyIR(float(duration_sec) * 0.75, (0.0, 0.0), -float(amplitude_deg), (1.0, 1.0), 0.0),
        JointTransformKeyIR(float(duration_sec), (0.0, 0.0), 0.0, (1.0, 1.0), 0.0),
    )
    track = build_joint_track(
        "TRACK:" + clip_hash[:16],
        clip.clip_id,
        joint.canonical_joint_id,
        keys,
        metadata={
            "joint_selection_policy": selection_policy,
            "selected_joint_skin_mass": selected_skin_mass,
            "authored_joint_name_dependency": False,
        },
    )
    state = build_motion_state(
        (clip,),
        (track,),
        metadata={
            "producer": "RealSaS.MotionCompiler.DeterministicPreset.v2",
            "authored_joint_names_used": False,
            "full_3d_motion_authority": False,
            "joint_selection_policy": selection_policy,
            "selected_joint_skin_mass": selected_skin_mass,
        },
    )
    validate_motion_against_mechanical(state, mechanical)
    return state
