from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_role_retarget_v1 import (
    INHERIT_PARENT,
    WORLD_ROTATION_DELTA,
    MotionRoleBindingV1IR,
    retarget_world_rotation_deltas_v1,
    validate_motion_role_bindings_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _matrix(*, angle_deg=0.0, translation=(0.0, 0.0, 0.0)):
    angle = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


def test_world_rotation_retarget_ignores_control_translation_and_preserves_target_lengths():
    source_ids = ("root", "leg", "foot.ctrl")
    source_rest = np.stack((
        _matrix(translation=(0.0, 0.0, 0.0)),
        _matrix(translation=(0.0, 0.0, 1.0)),
        _matrix(translation=(0.0, 0.0, 0.2)),
    ))
    source_pose = np.stack((
        source_rest,
        np.stack((
            _matrix(translation=(0.0, 0.0, 0.0)),
            _matrix(angle_deg=30.0, translation=(0.0, 0.0, 1.0)),
            _matrix(angle_deg=90.0, translation=(8.0, -5.0, 4.0)),
        )),
    ))

    target_ids = ("root", "leg", "foot", "socket")
    target_parents = np.asarray([-1, 0, 1, 2], dtype=np.int64)
    target_rest = np.stack((
        _matrix(translation=(0.0, 0.0, 0.0)),
        _matrix(translation=(0.0, 0.0, 2.0)),
        _matrix(translation=(0.0, 0.0, 3.0)),
        _matrix(translation=(0.0, 0.0, 3.5)),
    ))
    bindings = (
        MotionRoleBindingV1IR("root", "root", WORLD_ROTATION_DELTA),
        MotionRoleBindingV1IR("leg", "leg", WORLD_ROTATION_DELTA),
        MotionRoleBindingV1IR("foot", "foot.ctrl", WORLD_ROTATION_DELTA),
        MotionRoleBindingV1IR("socket", None, INHERIT_PARENT),
    )

    target_pose, report = retarget_world_rotation_deltas_v1(
        source_rest_global=source_rest,
        source_pose_global=source_pose,
        source_joint_ids=source_ids,
        target_rest_global=target_rest,
        target_joint_ids=target_ids,
        target_parent_indices=target_parents,
        bindings=bindings,
    )

    assert np.linalg.norm(target_pose[1, 2, :3, 3] - target_pose[1, 1, :3, 3]) == pytest.approx(1.0)
    assert np.linalg.norm(target_pose[1, 3, :3, 3] - target_pose[1, 2, :3, 3]) == pytest.approx(0.5)
    expected = _matrix(angle_deg=90.0)[:3, :3]
    assert np.allclose(target_pose[1, 2, :3, :3], expected)
    assert np.allclose(target_pose[1, 3, :3, :3], target_pose[1, 2, :3, :3])
    assert report["source_topology_required_to_match_target_topology"] is False
    assert report["max_bone_length_relative_residual"] < 1e-12


def test_root_world_translation_is_optional_and_scaled():
    source_rest = np.stack((_matrix(),))
    source_pose = np.stack((
        source_rest,
        np.stack((_matrix(translation=(2.0, 0.0, 0.0)),)),
    ))
    target_rest = np.stack((_matrix(translation=(10.0, 0.0, 0.0)),))
    bindings = (
        MotionRoleBindingV1IR(
            "root",
            "source.root",
            WORLD_ROTATION_DELTA,
            root_translation_from_source_world=True,
        ),
    )
    out, _ = retarget_world_rotation_deltas_v1(
        source_rest_global=source_rest,
        source_pose_global=source_pose,
        source_joint_ids=("source.root",),
        target_rest_global=target_rest,
        target_joint_ids=("root",),
        target_parent_indices=np.asarray([-1], dtype=np.int64),
        bindings=bindings,
        root_translation_scale=0.5,
    )
    assert np.allclose(out[1, 0, :3, 3], np.asarray([11.0, 0.0, 0.0]))


def test_validator_rejects_inherited_root():
    with pytest.raises(QualificationError, match="MOTION_ROLE_ROOT_MUST_BE_MAPPED"):
        validate_motion_role_bindings_v1(
            (MotionRoleBindingV1IR("root", None, INHERIT_PARENT),),
            source_joint_ids=("source.root",),
            target_joint_ids=("root",),
            target_parent_indices=np.asarray([-1], dtype=np.int64),
        )
