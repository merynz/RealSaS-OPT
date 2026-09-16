from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_3d_gates_v1 import (
    BilateralPairV1,
    HingeGuardSpecV1,
    half_period_bilateral_reflection_error_v1,
    loop_endpoint_errors_v1,
    max_hyperextension_degrees_v1,
    qualify_clip_gates_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_half_period_reflection_uses_time_not_array_half_and_handles_endpoint_inclusive_samples():
    times = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    positions = np.zeros((len(times), 3, 3), dtype=np.float64)
    for i, t in enumerate(times):
        s = np.sin(2.0 * np.pi * t)
        positions[i, 0] = [0.0, 0.0, 0.0]
        positions[i, 1] = [-1.0, s, 0.0]
        positions[i, 2] = [1.0, -s, 0.0]
    error = half_period_bilateral_reflection_error_v1(
        times,
        positions,
        root_index=0,
        bilateral_pairs=(BilateralPairV1(1, 2, "feet"),),
        anatomical_right=(1.0, 0.0, 0.0),
        body_height=2.0,
    )
    assert error <= 1e-12


def test_loop_gate_checks_position_and_endpoint_velocity():
    times = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    positions = np.zeros((len(times), 1, 3), dtype=np.float64)
    positions[:, 0, 1] = [0.0, 1.0, 0.0, -1.0, 0.0]
    pos_error, vel_error = loop_endpoint_errors_v1(times, positions, body_height=2.0)
    assert pos_error <= 1e-12
    assert vel_error <= 1e-12


def test_hyperextension_guard_uses_positive_flexion_sign_and_rejects_negative_bend():
    rest = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    positive = rest.copy()
    a = np.deg2rad(20.0)
    positive[2] = positive[1] + [np.cos(a), np.sin(a), 0.0]
    negative = rest.copy()
    b = np.deg2rad(-10.0)
    negative[2] = negative[1] + [np.cos(b), np.sin(b), 0.0]
    axes = np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float64)
    spec = HingeGuardSpecV1(
        name="knee",
        parent_index=0,
        joint_index=1,
        child_index=2,
        positive_flexion_axis_rest=(0.0, 0.0, 1.0),
        hyperextension_epsilon_degrees=2.0,
    )
    assert max_hyperextension_degrees_v1(rest, positive, (spec,), axes) <= 1e-12
    assert max_hyperextension_degrees_v1(rest, negative, (spec,), axes) == pytest.approx(8.0, abs=1e-9)


def test_clip_gate_requires_explicit_positive_preregistered_thresholds():
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    positions = np.zeros((3, 1, 3), dtype=np.float64)
    axes = np.asarray([[[0.0, 0.0, 1.0]]] * 3, dtype=np.float64)
    with pytest.raises(QualificationError, match="LOOP_POSITION_LIMIT_MUST_BE_EXPLICIT_POSITIVE"):
        qualify_clip_gates_v1(
            times=times,
            joint_positions=positions,
            rest_joint_positions=positions[0],
            joint_world_axes_by_frame=axes,
            body_height=1.0,
            hinge_specs=(),
            bilateral_pairs=(),
            root_index=0,
            anatomical_right=(1.0, 0.0, 0.0),
            loop_position_fraction_limit=0.0,
            loop_velocity_fraction_per_second_limit=0.1,
            bilateral_reflection_fraction_limit=0.1,
        )


def test_complete_synthetic_clip_gate_passes_with_locked_thresholds():
    times = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    positions = np.zeros((len(times), 3, 3), dtype=np.float64)
    for i, t in enumerate(times):
        s = np.sin(2.0 * np.pi * t)
        positions[i, 1] = [-1.0, s, 0.0]
        positions[i, 2] = [1.0, -s, 0.0]
    axes = np.zeros_like(positions)
    axes[:, :, 2] = 1.0
    report = qualify_clip_gates_v1(
        times=times,
        joint_positions=positions,
        rest_joint_positions=positions[0],
        joint_world_axes_by_frame=axes,
        body_height=2.0,
        hinge_specs=(),
        bilateral_pairs=(BilateralPairV1(1, 2, "feet"),),
        root_index=0,
        anatomical_right=(1.0, 0.0, 0.0),
        loop_position_fraction_limit=1.0e-6,
        loop_velocity_fraction_per_second_limit=1.0e-6,
        bilateral_reflection_fraction_limit=1.0e-6,
    )
    assert report.clip_gate_pass
    assert report.max_loop_position_fraction <= 1e-12
    assert report.max_loop_velocity_fraction_per_second <= 1e-12
    assert report.max_half_period_reflection_fraction <= 1e-12
