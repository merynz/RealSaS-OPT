from __future__ import annotations

"""Role-aware world-delta retargeting.

Source animation/control hierarchy is motion evidence, not target deform
hierarchy authority. Mapped target joints receive source world-rotation deltas
while target parentage and rest offsets remain authoritative. Unmapped
attachment/helper targets may inherit their parent with identity local delta.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .types import QualificationError


WORLD_ROTATION_DELTA = "WORLD_ROTATION_DELTA"
INHERIT_PARENT = "INHERIT_PARENT"
TRANSFER_MODES = {WORLD_ROTATION_DELTA, INHERIT_PARENT}


@dataclass(frozen=True)
class MotionRoleBindingV1IR:
    target_joint_id: str
    source_joint_id: str | None
    transfer_mode: str
    root_translation_from_source_world: bool = False
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _require_rigid_matrix(matrix: np.ndarray, code: str) -> None:
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (4, 4) or not np.isfinite(M).all():
        raise QualificationError(code)
    if float(np.max(np.abs(M[3] - np.asarray([0.0, 0.0, 0.0, 1.0])))) > 1.0e-9:
        raise QualificationError(code)
    R = M[:3, :3]
    if float(np.max(np.abs(R.T @ R - np.eye(3)))) > 1.0e-8 or float(np.linalg.det(R)) < 0.999999:
        raise QualificationError(code)


def _topological_order(parents: np.ndarray) -> tuple[int, ...]:
    p = np.asarray(parents, dtype=np.int64).reshape(-1)
    remaining = set(range(len(p)))
    out: list[int] = []
    while remaining:
        progressed = False
        for i in sorted(tuple(remaining)):
            parent = int(p[i])
            if parent < 0 or parent in out:
                out.append(i)
                remaining.remove(i)
                progressed = True
                break
        if not progressed:
            raise QualificationError("MOTION_ROLE_TARGET_SKELETON_CYCLE")
    return tuple(out)


def validate_motion_role_bindings_v1(
    bindings: Sequence[MotionRoleBindingV1IR],
    *,
    source_joint_ids: Sequence[str],
    target_joint_ids: Sequence[str],
    target_parent_indices: np.ndarray,
) -> None:
    source = {str(x) for x in source_joint_ids}
    target = tuple(map(str, target_joint_ids))
    parents = np.asarray(target_parent_indices, dtype=np.int64).reshape(-1)
    if len(target) == 0 or len(target) != len(parents) or len(target) != len(set(target)):
        raise QualificationError("MOTION_ROLE_TARGET_SKELETON_INVALID")
    if np.sum(parents < 0) != 1:
        raise QualificationError("MOTION_ROLE_TARGET_ROOT_INVALID")
    if any(int(p) >= len(target) for p in parents if int(p) >= 0):
        raise QualificationError("MOTION_ROLE_TARGET_PARENT_INVALID")
    _topological_order(parents)

    rows = tuple(bindings)
    if len(rows) != len(target):
        raise QualificationError("MOTION_ROLE_BINDING_TARGET_COVERAGE_INCOMPLETE")
    by_target = {str(row.target_joint_id): row for row in rows}
    if len(by_target) != len(rows) or set(by_target) != set(target):
        raise QualificationError("MOTION_ROLE_BINDING_TARGET_ID_INVALID")

    root_index = int(np.flatnonzero(parents < 0)[0])
    root_id = target[root_index]
    for target_id, row in by_target.items():
        mode = str(row.transfer_mode)
        if mode not in TRANSFER_MODES:
            raise QualificationError("MOTION_ROLE_TRANSFER_MODE_INVALID")
        source_id = None if row.source_joint_id in (None, "") else str(row.source_joint_id)
        if mode == WORLD_ROTATION_DELTA:
            if source_id is None or source_id not in source:
                raise QualificationError("MOTION_ROLE_SOURCE_JOINT_INVALID")
        elif source_id is not None:
            raise QualificationError("MOTION_ROLE_INHERIT_SOURCE_FORBIDDEN")
        if target_id == root_id and mode != WORLD_ROTATION_DELTA:
            raise QualificationError("MOTION_ROLE_ROOT_MUST_BE_MAPPED")
        if bool(row.root_translation_from_source_world) and target_id != root_id:
            raise QualificationError("MOTION_ROLE_ROOT_TRANSLATION_NONROOT_FORBIDDEN")


def retarget_world_rotation_deltas_v1(
    *,
    source_rest_global: np.ndarray,
    source_pose_global: np.ndarray,
    source_joint_ids: Sequence[str],
    target_rest_global: np.ndarray,
    target_joint_ids: Sequence[str],
    target_parent_indices: np.ndarray,
    bindings: Sequence[MotionRoleBindingV1IR],
    root_translation_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    source_rest = np.asarray(source_rest_global, dtype=np.float64)
    source_pose = np.asarray(source_pose_global, dtype=np.float64)
    target_rest = np.asarray(target_rest_global, dtype=np.float64)
    parents = np.asarray(target_parent_indices, dtype=np.int64).reshape(-1)

    if source_rest.ndim != 3 or source_rest.shape[1:] != (4, 4):
        raise QualificationError("MOTION_ROLE_SOURCE_REST_SHAPE_INVALID")
    if source_pose.ndim != 4 or source_pose.shape[1:] != source_rest.shape:
        raise QualificationError("MOTION_ROLE_SOURCE_POSE_SHAPE_INVALID")
    if target_rest.shape != (len(target_joint_ids), 4, 4):
        raise QualificationError("MOTION_ROLE_TARGET_REST_SHAPE_INVALID")
    if not np.isfinite(root_translation_scale) or float(root_translation_scale) <= 0.0:
        raise QualificationError("MOTION_ROLE_ROOT_TRANSLATION_SCALE_INVALID")
    for M in source_rest:
        _require_rigid_matrix(M, "MOTION_ROLE_SOURCE_REST_MATRIX_INVALID")
    for frame in source_pose:
        for M in frame:
            _require_rigid_matrix(M, "MOTION_ROLE_SOURCE_POSE_MATRIX_INVALID")
    for M in target_rest:
        _require_rigid_matrix(M, "MOTION_ROLE_TARGET_REST_MATRIX_INVALID")

    validate_motion_role_bindings_v1(
        bindings,
        source_joint_ids=source_joint_ids,
        target_joint_ids=target_joint_ids,
        target_parent_indices=parents,
    )
    source_index = {str(jid): i for i, jid in enumerate(source_joint_ids)}
    target_index = {str(jid): i for i, jid in enumerate(target_joint_ids)}
    by_target = {str(row.target_joint_id): row for row in bindings}
    order = _topological_order(parents)

    target_local_rest = np.empty_like(target_rest)
    for i, parent in enumerate(parents):
        target_local_rest[i] = target_rest[i] if int(parent) < 0 else np.linalg.inv(target_rest[int(parent)]) @ target_rest[i]

    out = np.empty((len(source_pose), len(target_joint_ids), 4, 4), dtype=np.float64)
    max_bone_length_relative_residual = 0.0
    max_rotation_orthogonality_residual = 0.0

    for frame_index, source_frame in enumerate(source_pose):
        target_frame = np.zeros_like(target_rest)
        target_frame[:, 3, 3] = 1.0

        desired_world_rotation: dict[int, np.ndarray] = {}
        for target_id, row in by_target.items():
            if row.transfer_mode != WORLD_ROTATION_DELTA:
                continue
            ti = target_index[target_id]
            si = source_index[str(row.source_joint_id)]
            R_delta = source_frame[si, :3, :3] @ source_rest[si, :3, :3].T
            desired_world_rotation[ti] = R_delta @ target_rest[ti, :3, :3]

        for ti in order:
            parent = int(parents[ti])
            row = by_target[str(target_joint_ids[ti])]
            if parent < 0:
                target_frame[ti, :3, :3] = desired_world_rotation[ti]
                target_frame[ti, :3, 3] = target_rest[ti, :3, 3]
                if row.root_translation_from_source_world:
                    si = source_index[str(row.source_joint_id)]
                    delta = source_frame[si, :3, 3] - source_rest[si, :3, 3]
                    target_frame[ti, :3, 3] += float(root_translation_scale) * delta
            else:
                local_offset = target_local_rest[ti, :3, 3]
                target_frame[ti, :3, 3] = (
                    target_frame[parent, :3, :3] @ local_offset
                    + target_frame[parent, :3, 3]
                )
                if row.transfer_mode == WORLD_ROTATION_DELTA:
                    target_frame[ti, :3, :3] = desired_world_rotation[ti]
                else:
                    target_frame[ti, :3, :3] = (
                        target_frame[parent, :3, :3]
                        @ target_local_rest[ti, :3, :3]
                    )

                rest_length = float(np.linalg.norm(target_rest[ti, :3, 3] - target_rest[parent, :3, 3]))
                posed_length = float(np.linalg.norm(target_frame[ti, :3, 3] - target_frame[parent, :3, 3]))
                if rest_length > 1.0e-12:
                    max_bone_length_relative_residual = max(
                        max_bone_length_relative_residual,
                        abs(posed_length / rest_length - 1.0),
                    )

            R = target_frame[ti, :3, :3]
            max_rotation_orthogonality_residual = max(
                max_rotation_orthogonality_residual,
                float(np.max(np.abs(R.T @ R - np.eye(3)))),
            )
            _require_rigid_matrix(
                target_frame[ti],
                "MOTION_ROLE_TARGET_POSE_MATRIX_INVALID",
            )

        out[frame_index] = target_frame

    report = {
        "schema": "RealSaS.MotionRoleRetargetReport.v1",
        "source_joint_count": int(len(source_joint_ids)),
        "target_joint_count": int(len(target_joint_ids)),
        "frame_count": int(len(source_pose)),
        "source_topology_required_to_match_target_topology": False,
        "translation_policy": "TARGET_REST_OFFSETS_AUTHORITATIVE__ROOT_WORLD_DELTA_OPTIONAL",
        "rotation_policy": WORLD_ROTATION_DELTA,
        "max_bone_length_relative_residual": float(max_bone_length_relative_residual),
        "max_rotation_orthogonality_residual": float(max_rotation_orthogonality_residual),
        "bindings": [row.to_dict() for row in bindings],
    }
    return out, report
