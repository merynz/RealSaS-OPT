from __future__ import annotations

"""Generic clip-level qualification gates for canonical 3D motion.

All tolerances that could be tuned to a subject result are explicit required
arguments. Callers must preregister them before evaluating a subject clip.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


D1_CLIP_GATES_SCHEMA = "RealSaS.Motion3DClipGates.v1"


@dataclass(frozen=True)
class HingeGuardSpecV1:
    name: str
    parent_index: int
    joint_index: int
    child_index: int
    positive_flexion_axis_rest: tuple[float, float, float]
    hyperextension_epsilon_degrees: float


@dataclass(frozen=True)
class BilateralPairV1:
    left_index: int
    right_index: int
    name: str


@dataclass(frozen=True)
class D1ClipGateReportV1:
    max_loop_position_fraction: float
    max_loop_velocity_fraction_per_second: float
    worst_hyperextension_degrees: float
    max_half_period_reflection_fraction: float
    loop_position_pass: bool
    loop_velocity_pass: bool
    hyperextension_pass: bool
    bilateral_reflection_pass: bool
    clip_gate_pass: bool
    thresholds: Mapping[str, float]
    report_hash: str
    schema_version: str = D1_CLIP_GATES_SCHEMA


def _unit(value, *, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(f"D1_GATE_{label}_INVALID")
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise QualificationError(f"D1_GATE_{label}_DEGENERATE")
    return v / n


def _validate_trajectory(times, joint_positions) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(joint_positions, dtype=np.float64)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all():
        raise QualificationError("D1_GATE_TIMES_INVALID")
    if np.any(np.diff(t) <= 0.0):
        raise QualificationError("D1_GATE_TIMES_NOT_STRICTLY_INCREASING")
    if p.ndim != 3 or p.shape[0] != len(t) or p.shape[2] != 3 or not np.isfinite(p).all():
        raise QualificationError("D1_GATE_JOINT_TRAJECTORY_INVALID")
    return t, p


def _require_positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise QualificationError(f"D1_GATE_{label}_MUST_BE_EXPLICIT_POSITIVE")
    return value


def signed_hinge_angle_degrees(parent, joint, child, axis) -> float:
    """Signed bend from parent->joint direction to joint->child around axis."""

    a = np.asarray(joint, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    b = np.asarray(child, dtype=np.float64) - np.asarray(joint, dtype=np.float64)
    n = _unit(axis, label="HINGE_AXIS")
    a = a - np.dot(a, n) * n
    b = b - np.dot(b, n) * n
    a = _unit(a, label="HINGE_PARENT_VECTOR")
    b = _unit(b, label="HINGE_CHILD_VECTOR")
    sine = float(np.dot(n, np.cross(a, b)))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def max_hyperextension_degrees_v1(
    rest_joint_positions,
    posed_joint_positions,
    hinge_specs: Sequence[HingeGuardSpecV1],
    joint_world_axes,
) -> float:
    """Return largest forbidden extension beyond each rest hinge angle.

    Positive motion is defined as flexion by each D0/D1 axis contract. A negative
    signed delta beyond epsilon is hyperextension. joint_world_axes must be from
    the same posed FK state, so parent rotations cannot hide a sign error.
    """

    rest = np.asarray(rest_joint_positions, dtype=np.float64)
    posed = np.asarray(posed_joint_positions, dtype=np.float64)
    axes = np.asarray(joint_world_axes, dtype=np.float64)
    if rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise QualificationError("D1_GATE_HINGE_POSITION_SHAPE_DRIFT")
    if axes.shape != rest.shape:
        raise QualificationError("D1_GATE_HINGE_AXIS_SHAPE_DRIFT")
    worst = 0.0
    for spec in hinge_specs:
        eps = _require_positive(spec.hyperextension_epsilon_degrees, "HYPEREXTENSION_EPSILON")
        indices = (spec.parent_index, spec.joint_index, spec.child_index)
        if min(indices) < 0 or max(indices) >= len(rest):
            raise QualificationError("D1_GATE_HINGE_INDEX_OUT_OF_RANGE")
        rest_axis = _unit(spec.positive_flexion_axis_rest, label="HINGE_REST_AXIS")
        rest_angle = signed_hinge_angle_degrees(
            rest[spec.parent_index], rest[spec.joint_index], rest[spec.child_index], rest_axis
        )
        posed_angle = signed_hinge_angle_degrees(
            posed[spec.parent_index], posed[spec.joint_index], posed[spec.child_index], axes[spec.joint_index]
        )
        delta = _wrap_degrees(posed_angle - rest_angle)
        forbidden = max(0.0, -delta - eps)
        worst = max(worst, forbidden)
    return float(worst)


def loop_endpoint_errors_v1(times, joint_positions, *, body_height: float) -> tuple[float, float]:
    t, p = _validate_trajectory(times, joint_positions)
    h = _require_positive(body_height, "BODY_HEIGHT")
    position_error = float(np.linalg.norm(p[-1] - p[0], axis=1).max(initial=0.0)) / h
    v0 = (p[1] - p[0]) / float(t[1] - t[0])
    v1 = (p[-1] - p[-2]) / float(t[-1] - t[-2])
    velocity_error = float(np.linalg.norm(v1 - v0, axis=1).max(initial=0.0)) / h
    return position_error, velocity_error


def _interpolate_joint_frame(t: np.ndarray, p: np.ndarray, target: float) -> np.ndarray:
    if target < t[0] - 1.0e-12 or target > t[-1] + 1.0e-12:
        raise QualificationError("D1_GATE_INTERPOLATION_TARGET_OUT_OF_RANGE")
    if target <= t[0]:
        return p[0]
    if target >= t[-1]:
        return p[-1]
    upper = int(np.searchsorted(t, target, side="left"))
    if abs(float(t[upper] - target)) <= 1.0e-12:
        return p[upper]
    lower = upper - 1
    alpha = float((target - t[lower]) / (t[upper] - t[lower]))
    return p[lower] + (p[upper] - p[lower]) * alpha


def half_period_bilateral_reflection_error_v1(
    times,
    joint_positions,
    *,
    root_index: int,
    bilateral_pairs: Sequence[BilateralPairV1],
    anatomical_right,
    body_height: float,
) -> float:
    """Measure left/right reflection at true t + T/2, independent of sample grid.

    Endpoint-inclusive 30fps bakes therefore do not get a phase error merely
    because the frame array length is odd or not divisible by two.
    """

    t, p = _validate_trajectory(times, joint_positions)
    if root_index < 0 or root_index >= p.shape[1]:
        raise QualificationError("D1_GATE_BILATERAL_ROOT_INDEX_INVALID")
    h = _require_positive(body_height, "BODY_HEIGHT")
    right = _unit(anatomical_right, label="ANATOMICAL_RIGHT")
    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise QualificationError("D1_GATE_BILATERAL_DURATION_INVALID")
    half_time = duration * 0.5
    worst = 0.0

    def reflect(v: np.ndarray) -> np.ndarray:
        return v - 2.0 * np.dot(v, right) * right

    for pair in bilateral_pairs:
        if min(pair.left_index, pair.right_index) < 0 or max(pair.left_index, pair.right_index) >= p.shape[1]:
            raise QualificationError("D1_GATE_BILATERAL_INDEX_OUT_OF_RANGE")

    # Only evaluate the first half; partner frames are interpolated at +T/2.
    for i, sample_time in enumerate(t):
        target = float(sample_time + half_time)
        if target > float(t[-1]) + 1.0e-12:
            break
        partner = _interpolate_joint_frame(t, p, target)
        root_i = p[i, root_index]
        root_j = partner[root_index]
        for pair in bilateral_pairs:
            left_i = p[i, pair.left_index] - root_i
            right_j = partner[pair.right_index] - root_j
            right_i = p[i, pair.right_index] - root_i
            left_j = partner[pair.left_index] - root_j
            worst = max(worst, float(np.linalg.norm(left_i - reflect(right_j))) / h)
            worst = max(worst, float(np.linalg.norm(right_i - reflect(left_j))) / h)
    return float(worst)


def qualify_clip_gates_v1(
    *,
    times,
    joint_positions,
    rest_joint_positions,
    joint_world_axes_by_frame,
    body_height: float,
    hinge_specs: Sequence[HingeGuardSpecV1],
    bilateral_pairs: Sequence[BilateralPairV1],
    root_index: int,
    anatomical_right,
    loop_position_fraction_limit: float,
    loop_velocity_fraction_per_second_limit: float,
    bilateral_reflection_fraction_limit: float,
) -> D1ClipGateReportV1:
    """Evaluate preregistered D1 clip gates without choosing thresholds from data."""

    pos_limit = _require_positive(loop_position_fraction_limit, "LOOP_POSITION_LIMIT")
    vel_limit = _require_positive(loop_velocity_fraction_per_second_limit, "LOOP_VELOCITY_LIMIT")
    bilateral_limit = _require_positive(bilateral_reflection_fraction_limit, "BILATERAL_LIMIT")
    t, p = _validate_trajectory(times, joint_positions)
    axes = np.asarray(joint_world_axes_by_frame, dtype=np.float64)
    if axes.shape != p.shape or not np.isfinite(axes).all():
        raise QualificationError("D1_GATE_WORLD_AXIS_TRAJECTORY_INVALID")

    loop_pos, loop_vel = loop_endpoint_errors_v1(t, p, body_height=body_height)
    worst_hyper = 0.0
    for frame_index in range(len(t)):
        worst_hyper = max(
            worst_hyper,
            max_hyperextension_degrees_v1(
                rest_joint_positions,
                p[frame_index],
                hinge_specs,
                axes[frame_index],
            ),
        )
    bilateral = half_period_bilateral_reflection_error_v1(
        t,
        p,
        root_index=root_index,
        bilateral_pairs=bilateral_pairs,
        anatomical_right=anatomical_right,
        body_height=body_height,
    )

    loop_pos_pass = loop_pos <= pos_limit
    loop_vel_pass = loop_vel <= vel_limit
    hyper_pass = worst_hyper <= 1.0e-12
    bilateral_pass = bilateral <= bilateral_limit
    passed = loop_pos_pass and loop_vel_pass and hyper_pass and bilateral_pass
    thresholds = {
        "loop_position_fraction_limit": pos_limit,
        "loop_velocity_fraction_per_second_limit": vel_limit,
        "bilateral_reflection_fraction_limit": bilateral_limit,
    }
    payload = {
        "schema": D1_CLIP_GATES_SCHEMA,
        "max_loop_position_fraction": loop_pos,
        "max_loop_velocity_fraction_per_second": loop_vel,
        "worst_hyperextension_degrees": worst_hyper,
        "max_half_period_reflection_fraction": bilateral,
        "loop_position_pass": loop_pos_pass,
        "loop_velocity_pass": loop_vel_pass,
        "hyperextension_pass": hyper_pass,
        "bilateral_reflection_pass": bilateral_pass,
        "clip_gate_pass": passed,
        "thresholds": thresholds,
    }
    return D1ClipGateReportV1(
        max_loop_position_fraction=loop_pos,
        max_loop_velocity_fraction_per_second=loop_vel,
        worst_hyperextension_degrees=worst_hyper,
        max_half_period_reflection_fraction=bilateral,
        loop_position_pass=loop_pos_pass,
        loop_velocity_pass=loop_vel_pass,
        hyperextension_pass=hyper_pass,
        bilateral_reflection_pass=bilateral_pass,
        clip_gate_pass=passed,
        thresholds=thresholds,
        report_hash=content_sha256(payload),
    )
