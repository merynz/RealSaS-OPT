from __future__ import annotations

"""Generic adapter from qualified skeleton + frozen axis contract + MotionStateIR to D1.

The legacy motion IR supplies authored scalar rotations only. The frozen D0 axis
contract supplies the 3D meaning/sign. This adapter never infers a missing axis and
never upgrades non-rotation 2D transforms into 3D semantics implicitly.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_3d_v1 import D1JointSpec, D1Pose, solve_fk_v1
from compiler.realsas_compiler_core.types import QualificationError


D1_ADAPTER_SCHEMA = "RealSaS.Motion3DAdapter.v1"


@dataclass(frozen=True)
class D1AxisBindingV1:
    joint_id: str
    axis_rest: tuple[float, float, float]
    legacy_scalar_to_semantic_sign: float
    semantic_role: str


@dataclass(frozen=True)
class D1CompiledClipV1:
    clip_id: str
    duration_seconds: float
    loop: bool
    times: tuple[float, ...]
    semantic_rotation_degrees_by_frame: tuple[Mapping[str, float], ...]
    poses: tuple[D1Pose, ...]
    joint_ids: tuple[str, ...]
    axis_contract_hash: str
    source_motion_state_hash: str
    compile_hash: str
    schema_version: str = D1_ADAPTER_SCHEMA


def _unit_tuple(value, *, label: str) -> tuple[float, float, float]:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(f"D1_ADAPTER_{label}_INVALID")
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise QualificationError(f"D1_ADAPTER_{label}_DEGENERATE")
    return tuple(map(float, v / n))


def parse_axis_contract_v1(axis_contract: Mapping) -> tuple[dict[str, D1AxisBindingV1], str]:
    rows = tuple(axis_contract.get("joint_axes") or ())
    if not rows:
        raise QualificationError("D1_ADAPTER_AXIS_CONTRACT_EMPTY")
    bindings: dict[str, D1AxisBindingV1] = {}
    for row in rows:
        try:
            joint_id = str(row["canonical_joint_id"])
            axis = _unit_tuple(row["axis_xyz"], label="AXIS")
            sign = float(row.get("legacy_scalar_to_semantic_sign", 1.0))
            role = str(row.get("role") or "UNSPECIFIED")
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError("D1_ADAPTER_AXIS_ROW_INVALID") from exc
        if not joint_id or joint_id in bindings:
            raise QualificationError("D1_ADAPTER_AXIS_JOINT_DUPLICATE_OR_EMPTY")
        if not math.isfinite(sign) or abs(abs(sign) - 1.0) > 1.0e-9:
            raise QualificationError("D1_ADAPTER_LEGACY_SIGN_MUST_BE_PLUS_OR_MINUS_ONE")
        bindings[joint_id] = D1AxisBindingV1(joint_id, axis, sign, role)

    axis_hash = content_sha256({
        "schema": D1_ADAPTER_SCHEMA,
        "source_axis_schema": str(axis_contract.get("schema") or axis_contract.get("schema_version") or ""),
        "source_axis_status": str(axis_contract.get("status") or ""),
        "joint_axes": [
            {
                "joint_id": b.joint_id,
                "axis_rest": b.axis_rest,
                "legacy_scalar_to_semantic_sign": b.legacy_scalar_to_semantic_sign,
                "semantic_role": b.semantic_role,
            }
            for b in bindings.values()
        ],
    })
    return bindings, axis_hash


def build_joint_specs_v1(skeleton, axis_contract: Mapping) -> tuple[tuple[D1JointSpec, ...], dict[str, D1AxisBindingV1], str]:
    joints = tuple(getattr(skeleton, "joints", ()) or ())
    if not joints:
        raise QualificationError("D1_ADAPTER_SKELETON_EMPTY")
    joint_ids = tuple(str(j.canonical_joint_id) for j in joints)
    if len(set(joint_ids)) != len(joint_ids):
        raise QualificationError("D1_ADAPTER_SKELETON_JOINT_ID_DUPLICATE")
    index = {jid: i for i, jid in enumerate(joint_ids)}
    bindings, axis_hash = parse_axis_contract_v1(axis_contract)
    unknown_axes = set(bindings) - set(joint_ids)
    if unknown_axes:
        raise QualificationError(f"D1_ADAPTER_AXIS_REFERENCES_UNKNOWN_JOINT:{sorted(unknown_axes)}")

    specs: list[D1JointSpec] = []
    for i, joint in enumerate(joints):
        jid = joint_ids[i]
        parent_id = getattr(joint, "parent_canonical_id", None)
        if parent_id is None:
            parent_index = None
        else:
            parent_id = str(parent_id)
            if parent_id not in index:
                raise QualificationError(f"D1_ADAPTER_PARENT_UNKNOWN:{jid}:{parent_id}")
            parent_index = index[parent_id]
        position = tuple(map(float, getattr(joint, "position")))
        binding = bindings.get(jid)
        specs.append(D1JointSpec(
            joint_id=jid,
            parent_index=parent_index,
            rest_position=position,
            axis_rest=None if binding is None else binding.axis_rest,
        ))
    return tuple(specs), bindings, axis_hash


def _validate_rotation_only_key(key) -> None:
    translation = tuple(map(float, getattr(key, "translation_xy", (0.0, 0.0))))
    scale = tuple(map(float, getattr(key, "scale_xy", (1.0, 1.0))))
    depth = float(getattr(key, "depth_offset", 0.0))
    if len(translation) != 2 or max(abs(x) for x in translation) > 1.0e-12:
        raise QualificationError("D1_ADAPTER_TRANSLATION_SEMANTICS_NOT_QUALIFIED")
    if len(scale) != 2 or max(abs(x - 1.0) for x in scale) > 1.0e-12:
        raise QualificationError("D1_ADAPTER_SCALE_SEMANTICS_NOT_QUALIFIED")
    if abs(depth) > 1.0e-12:
        raise QualificationError("D1_ADAPTER_LEGACY_DEPTH_OFFSET_NOT_3D_Z")


def _sample_track_rotation(track, time_seconds: float) -> float:
    keys = tuple(getattr(track, "keys", ()) or ())
    if not keys:
        raise QualificationError("D1_ADAPTER_TRACK_KEYS_EMPTY")
    for key in keys:
        _validate_rotation_only_key(key)
    ordered = sorted(keys, key=lambda k: float(k.time_sec))
    times = np.asarray([float(k.time_sec) for k in ordered], dtype=np.float64)
    values = np.asarray([float(k.rotation_deg) for k in ordered], dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0.0):
        raise QualificationError("D1_ADAPTER_TRACK_KEYS_INVALID")
    t = float(time_seconds)
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    upper = int(np.searchsorted(times, t, side="left"))
    if abs(float(times[upper] - t)) <= 1.0e-12:
        return float(values[upper])
    lower = upper - 1
    alpha = float((t - times[lower]) / (times[upper] - times[lower]))
    return float(values[lower] + (values[upper] - values[lower]) * alpha)


def authored_clip_sample_times_v1(motion_state, clip_id: str) -> tuple[float, ...]:
    clip = next((c for c in motion_state.clips if str(c.clip_id) == str(clip_id)), None)
    if clip is None:
        raise QualificationError("D1_ADAPTER_CLIP_NOT_FOUND")
    duration = float(clip.duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise QualificationError("D1_ADAPTER_CLIP_DURATION_INVALID")
    values = {0.0, duration}
    for track in motion_state.joint_tracks:
        if str(track.clip_id) != str(clip_id):
            continue
        for key in track.keys:
            t = float(key.time_sec)
            if t < -1.0e-12 or t > duration + 1.0e-12:
                raise QualificationError("D1_ADAPTER_KEY_OUTSIDE_CLIP_DURATION")
            values.add(min(duration, max(0.0, t)))
    return tuple(sorted(values))


def compile_motion_clip_to_d1_v1(
    *,
    skeleton,
    axis_contract: Mapping,
    motion_state,
    clip_id: str,
    sample_times: Sequence[float] | None = None,
) -> D1CompiledClipV1:
    specs, bindings, axis_hash = build_joint_specs_v1(skeleton, axis_contract)
    joint_ids = tuple(j.joint_id for j in specs)
    clip = next((c for c in motion_state.clips if str(c.clip_id) == str(clip_id)), None)
    if clip is None:
        raise QualificationError("D1_ADAPTER_CLIP_NOT_FOUND")
    duration = float(clip.duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise QualificationError("D1_ADAPTER_CLIP_DURATION_INVALID")

    tracks = [t for t in motion_state.joint_tracks if str(t.clip_id) == str(clip_id)]
    track_by_joint = {str(t.canonical_joint_id): t for t in tracks}
    if len(track_by_joint) != len(tracks):
        raise QualificationError("D1_ADAPTER_DUPLICATE_JOINT_TRACK")
    unknown_tracks = set(track_by_joint) - set(joint_ids)
    if unknown_tracks:
        raise QualificationError(f"D1_ADAPTER_TRACK_REFERENCES_UNKNOWN_JOINT:{sorted(unknown_tracks)}")
    missing_axes = set(track_by_joint) - set(bindings)
    if missing_axes:
        # It is only safe to omit an axis if the track is identically zero.
        for jid in sorted(missing_axes):
            values = [float(k.rotation_deg) for k in track_by_joint[jid].keys]
            if any(abs(v) > 1.0e-12 for v in values):
                raise QualificationError(f"D1_ADAPTER_ANIMATED_JOINT_WITHOUT_FROZEN_AXIS:{jid}")

    if sample_times is None:
        times = authored_clip_sample_times_v1(motion_state, clip_id)
    else:
        times = tuple(float(x) for x in sample_times)
        if not times or any(not math.isfinite(x) for x in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise QualificationError("D1_ADAPTER_SAMPLE_TIMES_INVALID")
        if times[0] < -1.0e-12 or times[-1] > duration + 1.0e-12:
            raise QualificationError("D1_ADAPTER_SAMPLE_TIME_OUTSIDE_CLIP")

    semantic_frames: list[Mapping[str, float]] = []
    poses: list[D1Pose] = []
    for t in times:
        semantic: dict[str, float] = {}
        for jid, track in track_by_joint.items():
            legacy = _sample_track_rotation(track, t)
            binding = bindings.get(jid)
            if binding is None:
                if abs(legacy) > 1.0e-12:
                    raise QualificationError(f"D1_ADAPTER_ANIMATED_JOINT_WITHOUT_FROZEN_AXIS:{jid}")
                continue
            semantic[jid] = legacy * binding.legacy_scalar_to_semantic_sign
        semantic_frames.append(semantic)
        poses.append(solve_fk_v1(specs, semantic))

    motion_hash = str(getattr(motion_state, "motion_state_hash", "") or "")
    payload = {
        "schema": D1_ADAPTER_SCHEMA,
        "clip_id": str(clip_id),
        "duration_seconds": duration,
        "loop": bool(clip.loop),
        "times": list(times),
        "joint_ids": list(joint_ids),
        "axis_contract_hash": axis_hash,
        "source_motion_state_hash": motion_hash,
        "semantic_rotations": [dict(sorted(frame.items())) for frame in semantic_frames],
        "pose_hashes": [p.pose_hash for p in poses],
    }
    return D1CompiledClipV1(
        clip_id=str(clip_id),
        duration_seconds=duration,
        loop=bool(clip.loop),
        times=times,
        semantic_rotation_degrees_by_frame=tuple(semantic_frames),
        poses=tuple(poses),
        joint_ids=joint_ids,
        axis_contract_hash=axis_hash,
        source_motion_state_hash=motion_hash,
        compile_hash=content_sha256(payload),
    )
