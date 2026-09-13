from __future__ import annotations

import math

import numpy as np

from compiler.realsas_compiler_services.numerics.lbs import apply_lbs_probe_v1

from .deformation import mesh_skin_dense_weights
from .types import QualificationError


def _t(x: float, y: float) -> np.ndarray:
    return np.asarray(((1.0, 0.0, float(x)), (0.0, 1.0, float(y)), (0.0, 0.0, 1.0)), dtype=np.float64)


def _r(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _s(x: float, y: float) -> np.ndarray:
    return np.asarray(((float(x), 0.0, 0.0), (0.0, float(y), 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _identity_key_state():
    return (0.0, 0.0), 0.0, (1.0, 1.0), 0.0


def _state_is_exact_identity(state) -> bool:
    translation, rotation, scale, depth = state
    return (
        tuple(float(x) for x in translation) == (0.0, 0.0)
        and float(rotation) == 0.0
        and tuple(float(x) for x in scale) == (1.0, 1.0)
        and float(depth) == 0.0
    )


def _sample_track(track, t_sec: float):
    keys = track.keys
    if not keys:
        return _identity_key_state()
    t = float(t_sec)
    if t <= float(keys[0].time_sec):
        key = keys[0]
        return key.translation_xy, float(key.rotation_deg), key.scale_xy, float(key.depth_offset)
    if t >= float(keys[-1].time_sec):
        key = keys[-1]
        return key.translation_xy, float(key.rotation_deg), key.scale_xy, float(key.depth_offset)
    for left, right in zip(keys[:-1], keys[1:]):
        t0, t1 = float(left.time_sec), float(right.time_sec)
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            lerp = lambda a, b: float(a) + alpha * (float(b) - float(a))
            return (
                (lerp(left.translation_xy[0], right.translation_xy[0]), lerp(left.translation_xy[1], right.translation_xy[1])),
                lerp(left.rotation_deg, right.rotation_deg),
                (lerp(left.scale_xy[0], right.scale_xy[0]), lerp(left.scale_xy[1], right.scale_xy[1])),
                lerp(left.depth_offset, right.depth_offset),
            )
    raise QualificationError("MOTION_DEFORMATION_TRACK_SAMPLING_FAILED")


def _screen_affine_to_world_xz(value: np.ndarray) -> np.ndarray:
    """Embed screen-right=-world-X, screen-up=+world-Z affine into world XYZ."""
    a, c, e = map(float, value[0])
    b, d, f = map(float, value[1])
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = a
    out[0, 2] = -c
    out[0, 3] = -e
    out[2, 0] = -b
    out[2, 2] = d
    out[2, 3] = f
    return out


def motion_lbs_transforms(skeleton, motion_state, clip_id: str, time_sec: float) -> tuple[np.ndarray, tuple[str, ...]]:
    clips = {clip.clip_id: clip for clip in motion_state.clips}
    if clip_id not in clips:
        raise QualificationError("MOTION_DEFORMATION_UNKNOWN_CLIP")
    clip = clips[clip_id]
    duration = float(clip.duration_sec)
    t_sec = float(time_sec)
    if clip.loop and duration > 0.0 and t_sec > duration:
        t_sec = t_sec % duration
    if t_sec < 0.0 or t_sec > duration:
        raise QualificationError("MOTION_DEFORMATION_TIME_OUT_OF_RANGE")

    joints = tuple(sorted(skeleton.joints, key=lambda j: j.canonical_joint_id))
    if not joints:
        raise QualificationError("MOTION_DEFORMATION_REQUIRES_JOINTS")
    by_id = {j.canonical_joint_id: j for j in joints}
    tracks = {}
    for track in motion_state.joint_tracks:
        if track.clip_id != clip_id:
            continue
        if track.canonical_joint_id in tracks:
            raise QualificationError("MOTION_DEFORMATION_DUPLICATE_JOINT_TRACK")
        if track.canonical_joint_id not in by_id:
            raise QualificationError("MOTION_DEFORMATION_UNKNOWN_JOINT_TRACK")
        tracks[track.canonical_joint_id] = track

    sampled = {
        joint.canonical_joint_id: (
            _sample_track(tracks[joint.canonical_joint_id], t_sec)
            if joint.canonical_joint_id in tracks
            else _identity_key_state()
        )
        for joint in joints
    }

    # The authored rest frame and loop endpoint are product identity contracts, not
    # merely samples that happen to evaluate close to identity.  Returning literal
    # identity matrices here prevents rest-pose cancellation arithmetic from turning
    # an exact authority statement into a tolerance-dependent one.
    if all(_state_is_exact_identity(state) for state in sampled.values()):
        transforms = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(joints), axis=0)
        return transforms[None, ...], tuple(j.canonical_joint_id for j in joints)

    rest = {
        jid: np.asarray((-float(j.position[0]), float(j.position[2])), dtype=np.float64)
        for jid, j in by_id.items()
    }
    global_pose: dict[str, np.ndarray] = {}

    def solve(jid: str) -> np.ndarray:
        if jid in global_pose:
            return global_pose[jid]
        joint = by_id[jid]
        translation, rotation, scale, _depth = sampled[jid]
        local_anim = _t(float(translation[0]), float(translation[1])) @ _r(rotation) @ _s(scale[0], scale[1])
        parent = joint.parent_canonical_id
        if parent is None:
            value = _t(rest[jid][0], rest[jid][1]) @ local_anim
        else:
            if parent not in by_id:
                raise QualificationError("MOTION_DEFORMATION_PARENT_JOINT_MISSING")
            offset = rest[jid] - rest[parent]
            value = solve(parent) @ _t(offset[0], offset[1]) @ local_anim
        global_pose[jid] = value
        return value

    transforms = []
    for joint in joints:
        pose = solve(joint.canonical_joint_id)
        skin_screen = pose @ _t(-rest[joint.canonical_joint_id][0], -rest[joint.canonical_joint_id][1])
        transforms.append(_screen_affine_to_world_xz(skin_screen))
    return np.asarray(transforms, dtype=np.float64)[None, ...], tuple(j.canonical_joint_id for j in joints)


def verified_motion_deformation_report(mesh, mesh_skin, skeleton, motion_state, clip_id: str, *, sample_count: int = 17) -> dict:
    if int(sample_count) < 3:
        raise ValueError("sample_count must be >= 3")
    clips = {clip.clip_id: clip for clip in motion_state.clips}
    if clip_id not in clips:
        raise QualificationError("MOTION_DEFORMATION_UNKNOWN_CLIP")
    clip = clips[clip_id]
    points, weights, joint_order = mesh_skin_dense_weights(mesh, mesh_skin, skeleton)
    times = np.linspace(0.0, float(clip.duration_sec), int(sample_count), dtype=np.float64)
    frames = []
    displacement_rms = []
    displacement_max = []
    transform_exact_identity = []
    finite = True
    identity_bank = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(joint_order), axis=0)[None, ...]
    for t_sec in times:
        transforms, transform_joint_order = motion_lbs_transforms(skeleton, motion_state, clip_id, float(t_sec))
        if tuple(transform_joint_order) != tuple(joint_order):
            raise QualificationError("MOTION_DEFORMATION_JOINT_ORDER_MISMATCH")
        transform_exact_identity.append(bool(np.array_equal(transforms, identity_bank)))
        moved = apply_lbs_probe_v1(points, weights, transforms)[0].astype(np.float64)
        finite = finite and bool(np.isfinite(moved).all())
        displacement = np.linalg.norm(moved - points, axis=1)
        displacement_rms.append(float(np.sqrt(np.mean(displacement ** 2))))
        displacement_max.append(float(displacement.max(initial=0.0)))
        frames.append(moved)

    frame0 = np.linalg.norm(frames[0] - points, axis=1)
    loop = np.linalg.norm(frames[-1] - points, axis=1)
    frame0_max = float(frame0.max(initial=0.0))
    loop_max = float(loop.max(initial=0.0))
    peak_max = float(max(displacement_max, default=0.0))
    peak_rms = float(max(displacement_rms, default=0.0))

    # apply_lbs_probe_v1 intentionally returns float32.  Preserve an exact transform
    # authority gate, then judge the numerical rest-frame projection at a tolerance
    # derived from float32 machine epsilon and current coordinate magnitude.
    coord_scale = float(max(1.0, np.max(np.abs(points), initial=0.0)))
    float32_identity_tolerance = float(4.0 * np.finfo(np.float32).eps * coord_scale)
    frame0_transform_exact = bool(transform_exact_identity[0])
    loop_transform_exact = bool(transform_exact_identity[-1])
    frame0_numeric_pass = bool(frame0_max <= float32_identity_tolerance)
    loop_numeric_pass = bool(loop_max <= float32_identity_tolerance)
    dynamic_threshold = max(1.0e-7, 4.0 * float32_identity_tolerance)
    dynamic_nonzero = bool(peak_max > dynamic_threshold)
    status = "PASS" if (
        finite
        and frame0_transform_exact
        and loop_transform_exact
        and frame0_numeric_pass
        and loop_numeric_pass
        and dynamic_nonzero
    ) else "FAIL"
    return {
        "schema": "RealSaS.VerifiedMotionDeformationReport.v2",
        "status": status,
        "clip_id": clip_id,
        "duration_sec": float(clip.duration_sec),
        "sample_count": int(sample_count),
        "vertex_count": int(len(points)),
        "joint_count": int(len(joint_order)),
        "joint_order": list(joint_order),
        "frame0_max_displacement": frame0_max,
        "loop_closure_max_displacement": loop_max,
        "peak_rms_displacement": peak_rms,
        "peak_max_displacement": peak_max,
        "float32_identity_tolerance": float32_identity_tolerance,
        "dynamic_nonzero_threshold": float(dynamic_threshold),
        "finite": bool(finite),
        "frame0_transform_exact_identity_pass": frame0_transform_exact,
        "loop_transform_exact_identity_pass": loop_transform_exact,
        "frame0_numeric_projection_pass": frame0_numeric_pass,
        "loop_numeric_projection_pass": loop_numeric_pass,
        "frame0_exact_identity_pass": bool(frame0_transform_exact and frame0_numeric_pass),
        "loop_closure_identity_pass": bool(loop_transform_exact and loop_numeric_pass),
        "dynamic_nonzero_pass": dynamic_nonzero,
        "historical_mesh_authority_used": False,
        "historical_weight_authority_used": False,
    }
