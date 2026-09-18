from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_3d_v1 import (
    D1JointSpec,
    D1_AXIS_TRANSPORT,
    D1_ROTATION_COMPOSITION,
    D1_SIDE_VIEW_WITNESS_BODY_HEIGHT_FRACTION,
    apply_lbs_matrix_v1,
    apply_lbs_quaternion_reference_v1,
    max_bone_length_relative_error,
    project_joint_trajectories_v1,
    qualify_d1_implementation_v1,
    solve_fk_v1,
)


def _joints():
    return (
        D1JointSpec("root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        D1JointSpec("hip", 0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        D1JointSpec("knee", 1, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _camera(view_index: int):
    return {
        "view_index": view_index,
        "origin": [0.0, 0.0, -10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 4.0,
        "resolution": 8,
    }


def test_zero_fk_is_exact_bind_pose_and_skin_matrices_are_identity():
    joints = _joints()
    pose = solve_fk_v1(joints, {})
    rest = np.asarray([j.rest_position for j in joints], dtype=np.float64)
    assert pose.joint_positions == pytest.approx(rest, abs=1e-12)
    assert pose.joint_rotations == pytest.approx(np.repeat(np.eye(3)[None, :, :], 3, axis=0), abs=1e-12)
    assert pose.skin_matrices == pytest.approx(np.repeat(np.eye(4)[None, :, :], 3, axis=0), abs=1e-12)


def test_parent_rotation_transports_child_rest_axis_in_combined_pose():
    joints = _joints()
    pose = solve_fk_v1(joints, {"root": 90.0, "hip": 30.0})
    # hip rest axis +X must become +Y after root +90deg around +Z.
    assert pose.joint_world_axes[1] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    # hip location itself follows the parent/root rotation.
    assert pose.joint_positions[1] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert D1_AXIS_TRANSPORT == "AXIS_WORLD_EQUALS_PARENT_WORLD_ROTATION_TIMES_AXIS_REST"
    assert D1_ROTATION_COMPOSITION == "PARENT_WORLD_ROTATION_POSTMULTIPLY_REST_AXIS_ROTATION"


def test_fk_preserves_every_bone_length_under_combined_rotations():
    pose = solve_fk_v1(_joints(), {"root": 73.0, "hip": -41.0, "knee": 22.0})
    assert max_bone_length_relative_error(_joints(), pose) <= 1e-12


def test_matrix_and_quaternion_translation_lbs_are_independent_and_agree():
    joints = _joints()
    pose = solve_fk_v1(joints, {"root": 27.0, "hip": -38.0, "knee": 19.0})
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [1.8, -0.2, 0.3]],
        dtype=np.float64,
    )
    weights = np.asarray(
        [[1.0, 0.0, 0.0], [0.15, 0.70, 0.15], [0.0, 0.25, 0.75]],
        dtype=np.float64,
    )
    a = apply_lbs_matrix_v1(vertices, weights, pose.skin_matrices)
    b = apply_lbs_quaternion_reference_v1(vertices, weights, pose)
    assert a == pytest.approx(b, abs=1e-10)
    # Guard against accidentally making the reference path an alias.
    src = inspect.getsource(apply_lbs_quaternion_reference_v1)
    assert "scipy.spatial.transform" in src
    assert "apply_lbs_matrix_v1(" not in src


def test_d1_implementation_gate_locks_one_percent_side_view_tolerance_before_subject_results():
    joints = _joints()
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.eye(3, dtype=np.float64)
    report = qualify_d1_implementation_v1(
        joints,
        {"root": 15.0, "hip": -25.0, "knee": 35.0},
        vertices,
        weights,
    )
    assert report.implementation_pass
    assert report.zero_fk_identity_max_error <= 1e-12
    assert report.zero_lbs_identity_max_error <= 1e-12
    assert report.max_bone_length_relative_error <= 1e-12
    assert report.dual_lbs_max_error <= 1e-10
    assert report.side_view_witness_body_height_fraction == pytest.approx(0.01)
    assert D1_SIDE_VIEW_WITNESS_BODY_HEIGHT_FRACTION == pytest.approx(0.01)


def test_projected_joint_trajectories_keep_xy_and_depth_from_same_3d_pose():
    joints = _joints()
    poses = [
        solve_fk_v1(joints, {}),
        solve_fk_v1(joints, {"root": 90.0}),
    ]
    trajectories = project_joint_trajectories_v1(
        poses,
        {"V0": _camera(0)},
        required_view_ids=("V0",),
    )
    xyz = trajectories["V0"]
    assert xyz.shape == (2, 3, 3)
    # Root is stationary and remains at camera-forward depth 10.
    np.testing.assert_allclose(
        xyz[:, 0, :],
        np.asarray([[4.0, 4.0, 10.0], [4.0, 4.0, 10.0]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    # Hip moves from +X to +Y in world; screen Y is inverted.
    assert xyz[0, 1, :] == pytest.approx([5.0, 4.0, 10.0], abs=1e-12)
    assert xyz[1, 1, :] == pytest.approx([4.0, 3.0, 10.0], abs=1e-12)


def test_motion_3d_core_is_subject_agnostic():
    source = inspect.getsource(__import__(
        "compiler.realsas_compiler_core.motion_3d_v1",
        fromlist=["dummy"],
    )).lower()
    assert "mage" not in source
