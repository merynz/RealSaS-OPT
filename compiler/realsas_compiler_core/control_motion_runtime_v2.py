from __future__ import annotations

"""Deterministic runtime sampling for CanonicalMotionIR.v2.

Sampling yields animator-control transforms consumed by Mechanics.  It never
writes deform-joint tracks and never bypasses the rig constraint program.
"""

from typing import Sequence
import math

import numpy as np

from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_motion_ir_v2 import CanonicalMotionIRV2
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_control_solver_v1 import ControlTransformIRV1, MechanicsPoseIRV1, solve_control_pose_v1
from .types import QualificationError


def _normalize_quat(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(tuple(map(float, value)), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise QualificationError("MOTION_RUNTIME_QUAT_INVALID")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise QualificationError("MOTION_RUNTIME_QUAT_DEGENERATE")
    return q / norm


def _slerp(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, float, float, float]:
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = qa + float(alpha) * (qb - qa)
        q = q / float(np.linalg.norm(q))
    else:
        theta0 = math.acos(dot)
        sin0 = math.sin(theta0)
        theta = theta0 * float(alpha)
        s0 = math.sin(theta0 - theta) / sin0
        s1 = math.sin(theta) / sin0
        q = s0 * qa + s1 * qb
    return tuple(float(x) for x in q)  # type: ignore[return-value]


def _sample_time(motion: CanonicalMotionIRV2, time_seconds: float) -> float:
    t = float(time_seconds)
    duration = float(motion.duration_seconds)
    if not math.isfinite(t) or duration <= 0.0:
        raise QualificationError("MOTION_RUNTIME_TIME_INVALID")
    if motion.loop:
        if t == duration:
            return duration
        return t % duration
    return min(max(t, 0.0), duration)


def sample_control_transforms_v2(
    motion: CanonicalMotionIRV2,
    *,
    rig: CanonicalRigIRV2,
    mechanics: CanonicalMechanicsIRV1,
    time_seconds: float,
) -> tuple[ControlTransformIRV1, ...]:
    if motion.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("MOTION_RUNTIME_RIG_LINEAGE_DRIFT")
    if motion.control_program_hash != rig.control_program_hash:
        raise QualificationError("MOTION_RUNTIME_CONTROL_PROGRAM_DRIFT")
    if motion.mechanics_lineage_hash != mechanics.mechanics_lineage_hash:
        raise QualificationError("MOTION_RUNTIME_MECHANICS_LINEAGE_DRIFT")
    control_ids = {c.canonical_control_id for c in rig.controls}
    t = _sample_time(motion, time_seconds)
    out: list[ControlTransformIRV1] = []
    for track in motion.control_tracks:
        if track.canonical_control_id not in control_ids:
            raise QualificationError("MOTION_RUNTIME_CONTROL_UNKNOWN")
        keys = track.keys
        if not keys:
            raise QualificationError("MOTION_RUNTIME_TRACK_EMPTY")
        if t <= keys[0].time_seconds:
            left = right = keys[0]
            alpha = 0.0
        elif t >= keys[-1].time_seconds:
            left = right = keys[-1]
            alpha = 0.0
        else:
            left = keys[0]
            right = keys[-1]
            for index in range(len(keys) - 1):
                if keys[index].time_seconds <= t <= keys[index + 1].time_seconds:
                    left, right = keys[index], keys[index + 1]
                    break
            span = float(right.time_seconds - left.time_seconds)
            if span <= 0.0:
                raise QualificationError("MOTION_RUNTIME_KEY_ORDER_DRIFT")
            alpha = float((t - left.time_seconds) / span)
        translation = tuple(
            float((1.0 - alpha) * a + alpha * b)
            for a, b in zip(left.translation, right.translation)
        )
        scale = tuple(
            float((1.0 - alpha) * a + alpha * b)
            for a, b in zip(left.scale, right.scale)
        )
        rotation = _slerp(left.rotation_quat_wxyz, right.rotation_quat_wxyz, alpha)
        out.append(ControlTransformIRV1(
            canonical_control_id=track.canonical_control_id,
            translation=translation,  # type: ignore[arg-type]
            rotation_quat_wxyz=rotation,
            scale=scale,  # type: ignore[arg-type]
        ))
    return tuple(out)


def solve_motion_pose_v2(
    motion: CanonicalMotionIRV2,
    *,
    rig: CanonicalRigIRV2,
    mechanics: CanonicalMechanicsIRV1,
    time_seconds: float,
) -> MechanicsPoseIRV1:
    controls = sample_control_transforms_v2(
        motion,
        rig=rig,
        mechanics=mechanics,
        time_seconds=time_seconds,
    )
    return solve_control_pose_v1(rig, controls)


__all__ = ["sample_control_transforms_v2", "solve_motion_pose_v2"]
