from __future__ import annotations

"""Generic D1 canonical-3D motion implementation for RealSaS playback.

Axes are frozen in rest/anatomical coordinates and transported by the parent's
posed rotation. This module qualifies implementation mechanics; it does not claim
that an authored motion clip is semantically correct.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.types import QualificationError


D1_MOTION_3D_SCHEMA = "RealSaS.Motion3DImplementation.v1"
D1_ROTATION_COMPOSITION = "PARENT_WORLD_ROTATION_POSTMULTIPLY_REST_AXIS_ROTATION"
D1_AXIS_TRANSPORT = "AXIS_WORLD_EQUALS_PARENT_WORLD_ROTATION_TIMES_AXIS_REST"
D1_SIDE_VIEW_WITNESS_BODY_HEIGHT_FRACTION = 0.01
D1_ZERO_TOLERANCE = 1.0e-9
D1_BONE_LENGTH_RELATIVE_TOLERANCE = 1.0e-9
D1_DUAL_LBS_ABSOLUTE_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class D1JointSpec:
    joint_id: str
    parent_index: int | None
    rest_position: tuple[float, float, float]
    axis_rest: tuple[float, float, float] | None


@dataclass(frozen=True)
class D1Pose:
    joint_positions: np.ndarray
    joint_rotations: np.ndarray
    joint_world_axes: np.ndarray
    skin_matrices: np.ndarray
    pose_hash: str


@dataclass(frozen=True)
class D1ImplementationReport:
    zero_fk_identity_max_error: float
    zero_lbs_identity_max_error: float
    max_bone_length_relative_error: float
    dual_lbs_max_error: float
    side_view_witness_body_height_fraction: float
    rotation_composition: str
    axis_transport: str
    implementation_pass: bool
    report_hash: str
    schema_version: str = D1_MOTION_3D_SCHEMA


def _unit(value, *, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(f"D1_{label}_INVALID")
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise QualificationError(f"D1_{label}_DEGENERATE")
    return v / n


def _validate_joints(joints: Sequence[D1JointSpec]) -> tuple[D1JointSpec, ...]:
    rows = tuple(joints)
    if not rows:
        raise QualificationError("D1_JOINT_SET_EMPTY")
    if len({j.joint_id for j in rows}) != len(rows):
        raise QualificationError("D1_JOINT_ID_DUPLICATE")
    roots = [i for i, j in enumerate(rows) if j.parent_index is None]
    if len(roots) != 1:
        raise QualificationError("D1_EXACTLY_ONE_ROOT_REQUIRED")
    for i, joint in enumerate(rows):
        p = np.asarray(joint.rest_position, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise QualificationError("D1_REST_POSITION_INVALID")
        if joint.axis_rest is not None:
            _unit(joint.axis_rest, label="REST_AXIS")
        if joint.parent_index is not None:
            parent = int(joint.parent_index)
            if parent < 0 or parent >= len(rows) or parent == i:
                raise QualificationError("D1_PARENT_INDEX_INVALID")
    for start in range(len(rows)):
        seen: set[int] = set()
        cur: int | None = start
        while cur is not None:
            if cur in seen:
                raise QualificationError("D1_JOINT_HIERARCHY_CYCLE")
            seen.add(cur)
            cur = rows[cur].parent_index
    return rows


def _topological_order(joints: Sequence[D1JointSpec]) -> tuple[int, ...]:
    rows = _validate_joints(joints)
    children: dict[int, list[int]] = {i: [] for i in range(len(rows))}
    root = next(i for i, j in enumerate(rows) if j.parent_index is None)
    for i, joint in enumerate(rows):
        if joint.parent_index is not None:
            children[int(joint.parent_index)].append(i)
    for values in children.values():
        values.sort()
    order: list[int] = []
    stack = [root]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(reversed(children[i]))
    if len(order) != len(rows):
        raise QualificationError("D1_JOINT_HIERARCHY_DISCONNECTED")
    return tuple(order)


def axis_angle_matrix(axis, degrees: float) -> np.ndarray:
    a = _unit(axis, label="ROTATION_AXIS")
    angle = math.radians(float(degrees))
    if not math.isfinite(angle):
        raise QualificationError("D1_ROTATION_ANGLE_NONFINITE")
    x, y, z = map(float, a)
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
        ],
        dtype=np.float64,
    )


def solve_fk_v1(
    joints: Sequence[D1JointSpec],
    rotation_degrees_by_joint: Mapping[str, float],
) -> D1Pose:
    rows = _validate_joints(joints)
    unknown = set(rotation_degrees_by_joint) - {j.joint_id for j in rows}
    if unknown:
        raise QualificationError(f"D1_ROTATION_REFERENCES_UNKNOWN_JOINT:{sorted(unknown)}")
    rest = np.asarray([j.rest_position for j in rows], dtype=np.float64)
    positions = np.zeros_like(rest)
    rotations = np.zeros((len(rows), 3, 3), dtype=np.float64)
    world_axes = np.zeros((len(rows), 3), dtype=np.float64)

    for i in _topological_order(rows):
        joint = rows[i]
        theta = float(rotation_degrees_by_joint.get(joint.joint_id, 0.0))
        if not math.isfinite(theta):
            raise QualificationError("D1_ROTATION_ANGLE_NONFINITE")
        if joint.axis_rest is None:
            if abs(theta) > 1.0e-12:
                raise QualificationError(f"D1_ROTATION_REQUIRES_FROZEN_AXIS:{joint.joint_id}")
            axis_rest = None
            local_rotation = np.eye(3, dtype=np.float64)
        else:
            axis_rest = _unit(joint.axis_rest, label="REST_AXIS")
            local_rotation = axis_angle_matrix(axis_rest, theta)
        if joint.parent_index is None:
            parent_rotation = np.eye(3, dtype=np.float64)
            positions[i] = rest[i]
        else:
            parent = int(joint.parent_index)
            parent_rotation = rotations[parent]
            rest_offset = rest[i] - rest[parent]
            positions[i] = positions[parent] + parent_rotation @ rest_offset
        if axis_rest is not None:
            world_axes[i] = parent_rotation @ axis_rest
        rotations[i] = parent_rotation @ local_rotation

    skin = np.zeros((len(rows), 4, 4), dtype=np.float64)
    skin[:, 3, 3] = 1.0
    for i in range(len(rows)):
        skin[i, :3, :3] = rotations[i]
        skin[i, :3, 3] = positions[i] - rotations[i] @ rest[i]

    payload = {
        "schema": D1_MOTION_3D_SCHEMA,
        "rotation_composition": D1_ROTATION_COMPOSITION,
        "axis_transport": D1_AXIS_TRANSPORT,
        "joint_ids": [j.joint_id for j in rows],
        "angles": [float(rotation_degrees_by_joint.get(j.joint_id, 0.0)) for j in rows],
        "axis_present": [j.axis_rest is not None for j in rows],
        "positions": positions.tolist(),
        "rotations": rotations.tolist(),
    }
    return D1Pose(
        joint_positions=positions,
        joint_rotations=rotations,
        joint_world_axes=world_axes,
        skin_matrices=skin,
        pose_hash=content_sha256(payload),
    )


def _validate_skin_payload(vertices, weights, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(vertices, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("D1_LBS_VERTICES_INVALID")
    if w.shape != (len(p), int(joint_count)) or not np.isfinite(w).all():
        raise QualificationError("D1_LBS_WEIGHTS_INVALID")
    if np.any(w < -1.0e-12):
        raise QualificationError("D1_LBS_NEGATIVE_WEIGHT")
    row_sum = w.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1.0e-9, rtol=0.0):
        raise QualificationError("D1_LBS_WEIGHT_ROWS_MUST_SUM_TO_ONE")
    return p, w


def apply_lbs_matrix_v1(vertices, weights, skin_matrices) -> np.ndarray:
    skin = np.asarray(skin_matrices, dtype=np.float64)
    if skin.ndim != 3 or skin.shape[1:] != (4, 4) or not np.isfinite(skin).all():
        raise QualificationError("D1_SKIN_MATRIX_PAYLOAD_INVALID")
    p, w = _validate_skin_payload(vertices, weights, len(skin))
    homo = np.concatenate((p, np.ones((len(p), 1), dtype=np.float64)), axis=1)
    transformed = np.einsum("jab,nb->jna", skin, homo, optimize=True)[:, :, :3]
    out = np.einsum("nj,jna->na", w, transformed, optimize=True)
    if not np.isfinite(out).all():
        raise QualificationError("D1_MATRIX_LBS_NONFINITE")
    return out


def apply_lbs_quaternion_reference_v1(vertices, weights, pose: D1Pose) -> np.ndarray:
    """Independent LBS witness using scipy quaternion rotations + translation."""

    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise QualificationError("D1_SCIPY_ROTATION_REFERENCE_REQUIRED") from exc

    rotations = np.asarray(pose.joint_rotations, dtype=np.float64)
    skin = np.asarray(pose.skin_matrices, dtype=np.float64)
    p, w = _validate_skin_payload(vertices, weights, len(rotations))
    q = Rotation.from_matrix(rotations)
    translations = skin[:, :3, 3]
    out = np.zeros_like(p)
    for joint_index in range(len(rotations)):
        rotated = q[joint_index].apply(p)
        transformed = rotated + translations[joint_index][None, :]
        out += w[:, joint_index:joint_index+1] * transformed
    if not np.isfinite(out).all():
        raise QualificationError("D1_QUATERNION_LBS_NONFINITE")
    return out


def max_bone_length_relative_error(joints: Sequence[D1JointSpec], pose: D1Pose) -> float:
    rows = _validate_joints(joints)
    rest = np.asarray([j.rest_position for j in rows], dtype=np.float64)
    posed = np.asarray(pose.joint_positions, dtype=np.float64)
    worst = 0.0
    for i, joint in enumerate(rows):
        if joint.parent_index is None:
            continue
        parent = int(joint.parent_index)
        a = float(np.linalg.norm(rest[i] - rest[parent]))
        b = float(np.linalg.norm(posed[i] - posed[parent]))
        if a <= 1.0e-12:
            raise QualificationError("D1_ZERO_LENGTH_BONE")
        worst = max(worst, abs(b - a) / a)
    return float(worst)


def project_joint_trajectories_v1(
    poses: Sequence[D1Pose],
    cameras: Mapping[str, Mapping],
    *,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> dict[str, np.ndarray]:
    if not poses:
        raise QualificationError("D1_TRAJECTORY_POSE_SET_EMPTY")
    joint_count = len(poses[0].joint_positions)
    if any(np.asarray(p.joint_positions).shape != (joint_count, 3) for p in poses):
        raise QualificationError("D1_TRAJECTORY_JOINT_COUNT_DRIFT")
    view_ids = tuple(map(str, required_view_ids))
    if set(map(str, cameras.keys())) != set(view_ids):
        raise QualificationError("D1_TRAJECTORY_CAMERA_VIEW_SET_MISMATCH")
    out: dict[str, np.ndarray] = {}
    for view_index, view_id in enumerate(view_ids):
        camera: CameraProjectionV3 = qualify_camera_v3(
            cameras[view_id], view_id=view_id, view_index=view_index
        )
        rows = [project_points_xyz_v3(p.joint_positions, camera) for p in poses]
        out[view_id] = np.stack(rows, axis=0)
    return out


def qualify_d1_implementation_v1(
    joints: Sequence[D1JointSpec],
    sample_rotations: Mapping[str, float],
    sample_vertices,
    sample_weights,
) -> D1ImplementationReport:
    rows = _validate_joints(joints)
    zero = solve_fk_v1(rows, {})
    rest_joints = np.asarray([j.rest_position for j in rows], dtype=np.float64)
    zero_fk = float(np.max(np.abs(zero.joint_positions - rest_joints)))

    p, w = _validate_skin_payload(sample_vertices, sample_weights, len(rows))
    zero_lbs_points = apply_lbs_matrix_v1(p, w, zero.skin_matrices)
    zero_lbs = float(np.max(np.abs(zero_lbs_points - p)))

    pose = solve_fk_v1(rows, sample_rotations)
    bone_error = max_bone_length_relative_error(rows, pose)
    matrix_points = apply_lbs_matrix_v1(p, w, pose.skin_matrices)
    quaternion_points = apply_lbs_quaternion_reference_v1(p, w, pose)
    dual_error = float(np.max(np.abs(matrix_points - quaternion_points)))

    passed = (
        zero_fk <= D1_ZERO_TOLERANCE
        and zero_lbs <= D1_ZERO_TOLERANCE
        and bone_error <= D1_BONE_LENGTH_RELATIVE_TOLERANCE
        and dual_error <= D1_DUAL_LBS_ABSOLUTE_TOLERANCE
    )
    payload = {
        "schema": D1_MOTION_3D_SCHEMA,
        "zero_fk_identity_max_error": zero_fk,
        "zero_lbs_identity_max_error": zero_lbs,
        "max_bone_length_relative_error": bone_error,
        "dual_lbs_max_error": dual_error,
        "side_view_witness_body_height_fraction": D1_SIDE_VIEW_WITNESS_BODY_HEIGHT_FRACTION,
        "rotation_composition": D1_ROTATION_COMPOSITION,
        "axis_transport": D1_AXIS_TRANSPORT,
        "implementation_pass": passed,
    }
    return D1ImplementationReport(
        zero_fk_identity_max_error=zero_fk,
        zero_lbs_identity_max_error=zero_lbs,
        max_bone_length_relative_error=bone_error,
        dual_lbs_max_error=dual_error,
        side_view_witness_body_height_fraction=D1_SIDE_VIEW_WITNESS_BODY_HEIGHT_FRACTION,
        rotation_composition=D1_ROTATION_COMPOSITION,
        axis_transport=D1_AXIS_TRANSPORT,
        implementation_pass=passed,
        report_hash=content_sha256(payload),
    )
