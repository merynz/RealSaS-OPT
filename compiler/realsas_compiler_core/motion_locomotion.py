from __future__ import annotations

"""Recovered phase-aware locomotion authoring for the bounded Mage lane.

The previous bounded preset used mostly sinusoidal joint rotations and forced loop
boundaries to the neutral/rest pose. This recovery restores the historical SaS
principle that locomotion is phase-authored (plant/release/flight/contact) and that
loop correctness means cyclic-pose continuity, not mandatory return-to-rest.

Only qualified canonical joint roles are consumed. No source joint names, proposal
indices, mesh authority or skin ownership are inferred here.
"""

import math

from .hashing import content_sha256
from .motion_quality import hermite_sample
from .semantic_joint_roles import infer_humanoid_joint_roles
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR


PRODUCER = "RealSaS.MotionCompiler.HistoricalPhaseLocomotion.v1"
SCHEMA = "RealSaS.HistoricalPhaseLocomotion.v1"


def _periodic_curve(u: float, knots: tuple[tuple[float, float], ...], tension: float = 0.12) -> float:
    u = float(u) % 1.0
    times = tuple(float(t) for t, _ in knots)
    values = tuple(float(v) for _, v in knots)
    if abs(times[0]) > 1e-12 or abs(times[-1] - 1.0) > 1e-12 or abs(values[0] - values[-1]) > 1e-9:
        raise ValueError("periodic gait curve requires matching 0/1 endpoints")
    return hermite_sample(times, values, u, tension=tension)


_HIP = (
    (0.00, -8.0), (0.12, -20.0), (0.28, -4.0), (0.42, 20.0),
    (0.50, 16.0), (0.64, 4.0), (0.82, -14.0), (1.00, -8.0),
)
_KNEE = (
    (0.00, 8.0), (0.12, 5.0), (0.28, 30.0), (0.42, 22.0),
    (0.50, 10.0), (0.64, 6.0), (0.82, 10.0), (1.00, 8.0),
)
_ANKLE = (
    (0.00, -4.0), (0.12, 9.0), (0.28, -10.0), (0.42, -7.0),
    (0.50, -3.0), (0.64, 8.0), (0.82, -6.0), (1.00, -4.0),
)
_TOE = (
    (0.00, 2.0), (0.12, 8.0), (0.28, -4.0), (0.42, -2.0),
    (0.50, 2.0), (0.64, 7.0), (0.82, -2.0), (1.00, 2.0),
)


def _roles(mechanical):
    binding = infer_humanoid_joint_roles(mechanical.skeleton)
    by_id = {j.canonical_joint_id: j for j in mechanical.skeleton.joints}
    return binding, by_id


def _pose_run(u: float, binding, by_id):
    rotations = {jid: 0.0 for jid in by_id}
    left_phase = u % 1.0
    right_phase = (u + 0.5) % 1.0

    for phase, chain in ((left_phase, binding.left_leg_chain), (right_phase, binding.right_leg_chain)):
        if chain:
            rotations[chain[0]] = _periodic_curve(phase, _HIP)
        if len(chain) > 1:
            rotations[chain[1]] = _periodic_curve(phase, _KNEE)
        if len(chain) > 2:
            rotations[chain[2]] = _periodic_curve(phase, _ANKLE)
        if len(chain) > 3:
            rotations[chain[3]] = _periodic_curve(phase, _TOE)

    rotations[binding.root_joint_id] = 2.6 * math.sin(4.0 * math.pi * u)
    spine = tuple(binding.spine_chain)
    for i, jid in enumerate(spine):
        rotations[jid] = -(1.4 + 0.32 * i) * math.sin(2.0 * math.pi * u + 0.18 * i)
    rotations[binding.head_joint_id] = 1.1 * math.sin(2.0 * math.pi * u + 0.35)

    for side_index, chain in enumerate((binding.left_arm_chain, binding.right_arm_chain)):
        phase = (u + (0.5 if side_index == 0 else 0.0)) % 1.0
        swing = _periodic_curve(phase, _HIP) / 20.0
        side = -1.0 if side_index == 0 else 1.0
        if chain:
            rotations[chain[0]] = 16.0 * swing
        if len(chain) > 1:
            rotations[chain[1]] = side * (5.0 + 5.0 * max(0.0, swing))
        if len(chain) > 2:
            rotations[chain[2]] = -side * 3.5 * swing
        if len(chain) > 3:
            rotations[chain[3]] = -side * 1.8 * swing
    return rotations


def _pose_idle(u: float, binding, by_id):
    rotations = {jid: 0.0 for jid in by_id}
    phase = 2.0 * math.pi * u
    rotations[binding.root_joint_id] = 0.65 * math.sin(phase)
    for i, jid in enumerate(binding.spine_chain):
        rotations[jid] = (0.35 + 0.10 * i) * math.sin(phase - 0.16 * (i + 1))
    rotations[binding.head_joint_id] = -0.65 * math.sin(phase - 0.30)
    for side_index, chain in enumerate((binding.left_arm_chain, binding.right_arm_chain)):
        side = -1.0 if side_index == 0 else 1.0
        if chain:
            rotations[chain[0]] = side * 1.8 * math.sin(phase + side * 0.20)
        if len(chain) > 1:
            rotations[chain[1]] = side * 1.1 * math.sin(phase - 0.55)
        if len(chain) > 2:
            rotations[chain[2]] = -side * 0.8 * math.sin(phase - 0.18)
    return rotations


def _effective(keys) -> bool:
    first = float(keys[0].rotation_deg)
    return any(abs(float(key.rotation_deg) - first) > 1e-12 for key in keys[1:])


def build_mage_historical_phase_motion(mechanical, *, sample_count: int = 33):
    sample_count = int(sample_count)
    if sample_count < 9 or sample_count % 2 == 0:
        raise ValueError("historical phase motion sample_count must be odd and >=9")
    binding, by_id = _roles(mechanical)
    role_payload = {
        "root": binding.root_joint_id,
        "spine": list(binding.spine_chain),
        "head": binding.head_joint_id,
        "left_arm": list(binding.left_arm_chain),
        "right_arm": list(binding.right_arm_chain),
        "left_leg": list(binding.left_leg_chain),
        "right_leg": list(binding.right_leg_chain),
    }
    clips = []
    tracks = []
    rows = (
        ("idle", "mage_fit1_idle_v2", 2.0, _pose_idle, "REST_CYCLIC"),
        ("run", "mage_fit1_run_v2", 0.8, _pose_run, "CYCLIC_POSE_NOT_REST_RETURN"),
    )
    for kind, clip_id, duration, pose_fn, loop_semantics in rows:
        times = tuple(float(duration) * i / (sample_count - 1) for i in range(sample_count))
        poses = tuple(pose_fn(i / (sample_count - 1), binding, by_id) for i in range(sample_count))
        poses = tuple(poses[:-1]) + ({**poses[0]},)
        spec = {
            "schema": SCHEMA,
            "producer": PRODUCER,
            "clip_id": clip_id,
            "intent": kind.upper(),
            "duration_sec": duration,
            "sample_count": sample_count,
            "roles": role_payload,
            "loop_cycle_semantics": loop_semantics,
            "phase_model": (
                "BREATH_FOLLOW_THROUGH" if kind == "idle"
                else "PLANT_RELEASE_FLIGHT_CONTACT_WITH_KNEE_COMPRESSION_TOE_ROLL"
            ),
            "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
        }
        clip_hash = content_sha256(spec)
        clips.append(MotionClipIR(
            clip_id,
            "PRESET",
            clip_hash,
            ("PRESET_MOTION", "WEIGHTED_DEFORM_2D_2P5D"),
            source_ref=PRODUCER,
            duration_sec=float(duration),
            loop=True,
            metadata={
                "intent": kind.upper(),
                "display_name": "Idle Historical" if kind == "idle" else "Run Historical",
                "spec_hash": clip_hash,
                "loop_cycle_semantics": loop_semantics,
                "exact_loop_pose_authored": True,
                "return_to_rest_required": kind == "idle",
                "contact_schedule_present": kind == "run",
                "contact_lock_claimed": False,
                "phase_model": spec["phase_model"],
                "semantic_joint_names_used": False,
                "historical_mesh_authority_used": False,
                "historical_weight_authority_used": False,
            },
        ))
        for jid in sorted(by_id):
            keys = tuple(
                JointTransformKeyIR(float(t), (0.0, 0.0), float(pose[jid]), (1.0, 1.0), 0.0)
                for t, pose in zip(times, poses)
            )
            if not _effective(keys):
                continue
            tracks.append(build_joint_track(
                f"TRACK:HISTORICAL:{clip_id}:{jid}",
                clip_id,
                jid,
                keys,
                metadata={
                    "producer": PRODUCER,
                    "phase_authored": True,
                    "loop_cycle_semantics": loop_semantics,
                    "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
                    "authored_joint_name_dependency": False,
                },
            ))
    state = build_motion_state(
        tuple(clips),
        tuple(tracks),
        metadata={
            "producer": PRODUCER,
            "historical_motion_engine_recovered": True,
            "preset_family": "MAGE_FIT1_HISTORICAL_PHASE_IDLE_RUN",
            "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
            "phase_authored_locomotion": True,
            "contact_lock_claimed": False,
            "xpbd_secondary_claimed": False,
            "corrective_deformation_claimed": False,
            "roles": role_payload,
        },
    )
    validate_motion_against_mechanical(state, mechanical)
    return state


__all__ = ["PRODUCER", "build_mage_historical_phase_motion"]
