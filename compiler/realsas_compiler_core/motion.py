from __future__ import annotations

import math

from .hashing import content_sha256
from .types import QualificationError
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR


_IDENTITY_TRANSLATION = (0.0, 0.0)
_IDENTITY_SCALE = (1.0, 1.0)


def _identity_key(time_sec: float) -> JointTransformKeyIR:
    return JointTransformKeyIR(float(time_sec), _IDENTITY_TRANSLATION, 0.0, _IDENTITY_SCALE, 0.0)


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
    joint = sorted(joints, key=lambda j: (-mass[j.canonical_joint_id], j.canonical_joint_id))[0]
    if mass[joint.canonical_joint_id] <= 0.0:
        raise QualificationError("MOTION_NO_SKIN_SUPPORTED_CANONICAL_JOINT")
    return joint, float(mass[joint.canonical_joint_id])


def build_deterministic_preset_motion(
    mechanical,
    *,
    clip_id: str = "preset_idle_v1",
    duration_sec: float = 1.0,
    amplitude_deg: float = 4.0,
):
    """Legacy-small deterministic preset retained as a generic fail-safe."""
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not (0.0 < abs(float(amplitude_deg)) <= 45.0):
        raise ValueError("bounded nonzero amplitude required")
    joint, selected_skin_mass = _select_preset_joint(mechanical)
    policy = "MAX_QUALIFIED_SKIN_MASS_THEN_CANONICAL_ID"
    spec = {
        "schema": "RealSaS.DeterministicPresetMotion.v2",
        "clip_id": clip_id,
        "duration_sec": float(duration_sec),
        "joint_id": joint.canonical_joint_id,
        "joint_parent": joint.parent_canonical_id,
        "selected_joint_skin_mass": selected_skin_mass,
        "amplitude_deg": float(amplitude_deg),
        "transform_space": "PUPPET_LOCAL_2D_2P5D",
        "selection_policy": policy,
    }
    clip_hash = content_sha256(spec)
    clip = MotionClipIR(
        str(clip_id), "PRESET", clip_hash, ("PRESET_MOTION",),
        source_ref="RealSaS.MotionCompiler.DeterministicPreset.v2",
        duration_sec=float(duration_sec), loop=True,
        metadata={
            "semantic_joint_name_used": False,
            "quaternion_used": False,
            "vec3_rigid_motion_used": False,
            "spec_hash": clip_hash,
            "joint_selection_policy": policy,
            "selected_joint_skin_mass": selected_skin_mass,
        },
    )
    keys = (
        _identity_key(0.0),
        JointTransformKeyIR(float(duration_sec) * 0.25, _IDENTITY_TRANSLATION, float(amplitude_deg), _IDENTITY_SCALE, 0.0),
        JointTransformKeyIR(float(duration_sec) * 0.75, _IDENTITY_TRANSLATION, -float(amplitude_deg), _IDENTITY_SCALE, 0.0),
        _identity_key(float(duration_sec)),
    )
    track = build_joint_track(
        "TRACK:" + clip_hash[:16], clip.clip_id, joint.canonical_joint_id, keys,
        metadata={
            "joint_selection_policy": policy,
            "selected_joint_skin_mass": selected_skin_mass,
            "authored_joint_name_dependency": False,
        },
    )
    state = build_motion_state(
        (clip,), (track,),
        metadata={
            "producer": "RealSaS.MotionCompiler.DeterministicPreset.v2",
            "authored_joint_names_used": False,
            "full_3d_motion_authority": False,
            "joint_selection_policy": policy,
            "selected_joint_skin_mass": selected_skin_mass,
        },
    )
    validate_motion_against_mechanical(state, mechanical)
    return state


def _children(joints):
    by_id = {j.canonical_joint_id: j for j in joints}
    out = {jid: [] for jid in by_id}
    for joint in joints:
        parent = joint.parent_canonical_id
        if parent is not None:
            if parent not in by_id:
                raise QualificationError("MAGE_PRESET_PARENT_JOINT_MISSING")
            out[parent].append(joint.canonical_joint_id)
    for rows in out.values():
        rows.sort()
    return out, by_id


def _chain_from(start, children):
    chain = [start]
    current = start
    while len(children[current]) == 1:
        current = children[current][0]
        chain.append(current)
    return tuple(chain)


def _infer_mage_topology(mechanical):
    """Infer Mage control roles from canonical topology/rest geometry only."""
    joints = tuple(mechanical.skeleton.joints)
    if len(joints) < 7:
        raise QualificationError("MAGE_PRESET_REQUIRES_HUMANOID_TOPOLOGY")
    children, by_id = _children(joints)
    roots = tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    declared = tuple(getattr(mechanical.skeleton, "deform_root_ids", ()))
    root_candidates = tuple(x for x in declared if x in by_id) or roots
    if len(root_candidates) != 1:
        raise QualificationError("MAGE_PRESET_REQUIRES_SINGLE_DEFORM_ROOT")
    root = root_candidates[0]
    root_pos = by_id[root].position
    root_children = tuple(children[root])
    if len(root_children) < 3:
        raise QualificationError("MAGE_PRESET_REQUIRES_SPINE_AND_TWO_LEGS")

    def central_up_score(jid):
        p = by_id[jid].position
        dz = float(p[2]) - float(root_pos[2])
        return (0 if dz > 0.0 else 1, abs(float(p[0]) - float(root_pos[0])), -dz, jid)

    spine_start = min(root_children, key=central_up_score)
    if float(by_id[spine_start].position[2]) <= float(root_pos[2]):
        raise QualificationError("MAGE_PRESET_CANNOT_INFER_UPWARD_SPINE")
    legs = sorted(
        (jid for jid in root_children if jid != spine_start),
        key=lambda jid: (float(by_id[jid].position[0]), jid),
    )
    if len(legs) < 2:
        raise QualificationError("MAGE_PRESET_REQUIRES_TWO_LEG_BRANCHES")
    legs = legs[:2]

    spine = [spine_start]
    current = spine_start
    while len(children[current]) == 1:
        nxt = children[current][0]
        if float(by_id[nxt].position[2]) <= float(by_id[current].position[2]):
            break
        spine.append(nxt)
        current = nxt
    chest = spine[-1]
    chest_children = tuple(children[chest])
    if len(chest_children) < 3:
        raise QualificationError("MAGE_PRESET_REQUIRES_HEAD_AND_TWO_ARM_BRANCHES")
    chest_pos = by_id[chest].position
    head = min(
        chest_children,
        key=lambda jid: (
            abs(float(by_id[jid].position[0]) - float(chest_pos[0])),
            -float(by_id[jid].position[2]), jid,
        ),
    )
    arms = sorted(
        (jid for jid in chest_children if jid != head),
        key=lambda jid: (float(by_id[jid].position[0]), jid),
    )
    if len(arms) < 2:
        raise QualificationError("MAGE_PRESET_REQUIRES_TWO_ARM_BRANCHES")
    arms = arms[:2]
    return {
        "root": root,
        "spine": tuple(spine),
        "chest": chest,
        "head": head,
        "legs": tuple(_chain_from(jid, children) for jid in legs),
        "arms": tuple(_chain_from(jid, children) for jid in arms),
    }, by_id


def _phase_delta(phase: float, offset: float) -> float:
    return math.sin(phase + offset) - math.sin(offset)


def _mage_pose(kind: str, phase: float, roles, by_id):
    state = {jid: (_IDENTITY_TRANSLATION, 0.0) for jid in by_id}
    root = roles["root"]
    if kind == "idle":
        state[root] = ((0.0, 0.004 * (1.0 - math.cos(2.0 * phase))), 0.8 * math.sin(phase))
        for i, jid in enumerate(roles["spine"]):
            state[jid] = (_IDENTITY_TRANSLATION, (0.45 + 0.12 * i) * _phase_delta(phase, 0.12 * (i + 1)))
        state[roles["head"]] = (_IDENTITY_TRANSLATION, -0.8 * _phase_delta(phase, 0.2))
        for side_index, chain in enumerate(roles["arms"]):
            side = -1.0 if side_index == 0 else 1.0
            if chain:
                state[chain[0]] = (_IDENTITY_TRANSLATION, side * 2.6 * _phase_delta(phase, side * 0.35))
            if len(chain) > 1:
                state[chain[1]] = (_IDENTITY_TRANSLATION, side * 2.0 * _phase_delta(phase, 0.8))
            if len(chain) > 2:
                state[chain[2]] = (_IDENTITY_TRANSLATION, -side * 1.5 * _phase_delta(phase, 0.2))
        for side_index, chain in enumerate(roles["legs"]):
            side = -1.0 if side_index == 0 else 1.0
            if chain:
                state[chain[0]] = (_IDENTITY_TRANSLATION, side * 1.4 * math.sin(phase))
            if len(chain) > 1:
                state[chain[1]] = (_IDENTITY_TRANSLATION, -side * 0.8 * math.sin(phase))
    elif kind == "run":
        state[root] = ((0.0, 0.014 * (1.0 - math.cos(2.0 * phase))), 3.0 * math.sin(2.0 * phase))
        for i, jid in enumerate(roles["spine"]):
            state[jid] = (_IDENTITY_TRANSLATION, (1.5 + 0.35 * i) * _phase_delta(2.0 * phase, 0.10 * (i + 1)))
        state[roles["head"]] = (_IDENTITY_TRANSLATION, -1.6 * _phase_delta(2.0 * phase, 0.25))
        for side_index, chain in enumerate(roles["legs"]):
            ph = phase + (0.0 if side_index == 0 else math.pi)
            swing = math.sin(ph)
            if chain:
                state[chain[0]] = (_IDENTITY_TRANSLATION, 30.0 * swing)
            if len(chain) > 1:
                state[chain[1]] = (_IDENTITY_TRANSLATION, 25.0 * max(0.0, -swing))
            if len(chain) > 2:
                state[chain[2]] = (_IDENTITY_TRANSLATION, -15.0 * max(0.0, -swing) + 5.0 * math.sin(2.0 * ph))
            if len(chain) > 3:
                state[chain[3]] = (_IDENTITY_TRANSLATION, 5.0 * math.sin(ph))
        for side_index, chain in enumerate(roles["arms"]):
            ph = phase + (math.pi if side_index == 0 else 0.0)
            swing = math.sin(ph)
            side = -1.0 if side_index == 0 else 1.0
            if chain:
                state[chain[0]] = (_IDENTITY_TRANSLATION, 22.0 * swing)
            if len(chain) > 1:
                state[chain[1]] = (_IDENTITY_TRANSLATION, side * 9.0 * swing)
            if len(chain) > 2:
                state[chain[2]] = (_IDENTITY_TRANSLATION, -side * 6.0 * swing)
    else:
        raise ValueError(kind)
    return state


def _is_effective(keys) -> bool:
    first = keys[0]
    baseline = (first.translation_xy, first.rotation_deg, first.scale_xy, first.depth_offset)
    return any((k.translation_xy, k.rotation_deg, k.scale_xy, k.depth_offset) != baseline for k in keys[1:])


def build_mage_topology_preset_motion(mechanical, *, sample_count: int = 17):
    """Build identity-safe Mage idle/run from qualified topology/rest geometry.

    Endpoint identity is assigned explicitly, not obtained from sin(2*pi), so the
    frame-0 and loop-closure contract is bit-stable rather than tolerance-authored.
    """
    sample_count = int(sample_count)
    if sample_count < 5 or sample_count % 2 == 0:
        raise ValueError("sample_count must be odd and >= 5")
    roles, by_id = _infer_mage_topology(mechanical)
    role_payload = {
        "root": roles["root"],
        "spine": list(roles["spine"]),
        "head": roles["head"],
        "arms": [list(x) for x in roles["arms"]],
        "legs": [list(x) for x in roles["legs"]],
    }
    clips = []
    tracks = []
    for kind, clip_id, duration in (("idle", "mage_fit1_idle_v2", 2.0), ("run", "mage_fit1_run_v2", 0.8)):
        times = tuple(float(duration) * i / (sample_count - 1) for i in range(sample_count))
        identity_pose = {jid: (_IDENTITY_TRANSLATION, 0.0) for jid in by_id}
        poses = []
        for index, time_sec in enumerate(times):
            if index in (0, sample_count - 1):
                poses.append(identity_pose)
            else:
                poses.append(_mage_pose(kind, 2.0 * math.pi * time_sec / float(duration), roles, by_id))
        spec = {
            "schema": "RealSaS.MageTopologyPresetMotion.v1",
            "clip_id": clip_id,
            "intent": kind.upper(),
            "duration_sec": float(duration),
            "sample_count": sample_count,
            "roles": role_payload,
            "frame0_identity": "EXPLICIT_EXACT",
            "loop_closure_identity": "EXPLICIT_EXACT",
            "transform_space": "PUPPET_LOCAL_2D_2P5D",
        }
        clip_hash = content_sha256(spec)
        clips.append(MotionClipIR(
            clip_id, "PRESET", clip_hash, ("PRESET_MOTION", "WEIGHTED_DEFORM_2D_2P5D"),
            source_ref="RealSaS.MotionCompiler.MageTopologyPreset.v1",
            duration_sec=float(duration), loop=True,
            metadata={
                "intent": kind.upper(), "display_name": kind.title(), "spec_hash": clip_hash,
                "semantic_joint_names_used": False,
                "historical_mesh_authority_used": False,
                "historical_weight_authority_used": False,
                "frame0_identity_authored": True,
                "loop_closure_identity_authored": True,
            },
        ))
        for jid in sorted(by_id):
            keys = tuple(
                JointTransformKeyIR(
                    float(t), tuple(map(float, pose[jid][0])), float(pose[jid][1]),
                    _IDENTITY_SCALE, 0.0,
                )
                for t, pose in zip(times, poses)
            )
            if _is_effective(keys):
                tracks.append(build_joint_track(
                    f"TRACK:{clip_id}:{jid}", clip_id, jid, keys,
                    metadata={
                        "producer": "RealSaS.MotionCompiler.MageTopologyPreset.v1",
                        "topology_role_inference": True,
                        "authored_joint_name_dependency": False,
                        "frame0_identity_authored": True,
                        "loop_closure_identity_authored": True,
                    },
                ))
    state = build_motion_state(
        tuple(clips), tuple(tracks),
        metadata={
            "producer": "RealSaS.MotionCompiler.MageTopologyPreset.v1",
            "preset_family": "MAGE_FIT1_IDLE_RUN",
            "authored_joint_names_used": False,
            "historical_mesh_authority_used": False,
            "historical_weight_authority_used": False,
            "frame0_identity_authored": True,
            "loop_closure_identity_authored": True,
            "roles": role_payload,
        },
    )
    validate_motion_against_mechanical(state, mechanical)
    return state
