from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from compiler.realsas_compiler_core.dynamic_appearance_conditioning_v2 import (
    dynamic_face_conditioning_metrics,
    validate_dynamic_appearance_policy,
)


ROOT = Path(__file__).resolve().parents[2]


def _policy():
    payload = json.loads(
        (
            ROOT
            / "canonical"
            / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
        ).read_text(encoding="utf-8")
    )
    return payload["completion_quality_policy"]


def test_dynamic_appearance_identity_map_is_well_conditioned():
    triangle = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    uv = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    metrics = dynamic_face_conditioning_metrics(
        uv_triangle=uv,
        posed_xyz_triangle=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        rest_xyz_triangle=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        posed_screen_triangle=triangle,
        previous_xyz_triangle=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        min_projected_double_area_px2=1.0,
    )
    assert metrics["measurable"] is True
    assert metrics["uv_to_surface_condition_number"] == pytest.approx(1.0)
    assert metrics["relative_surface_condition_number"] == pytest.approx(1.0)
    assert metrics["relative_surface_principal_stretch"] == pytest.approx(1.0)
    assert metrics["adjacent_frame_surface_principal_stretch"] == pytest.approx(1.0)


def test_rigid_3d_rotation_does_not_count_as_intrinsic_art_deformation():
    uv = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    rest_xyz = (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
    )
    # Rigid 60-degree rotation around Y. Intrinsic edge lengths are unchanged,
    # while an orthographic XY projection is strongly foreshortened in X.
    posed_xyz = (
        (0.0, 0.0, 0.0),
        (5.0, 0.0, -8.660254037844386),
        (0.0, 10.0, 0.0),
    )
    posed_screen = (
        (0.0, 0.0),
        (5.0, 0.0),
        (0.0, 10.0),
    )
    metrics = dynamic_face_conditioning_metrics(
        uv_triangle=uv,
        posed_xyz_triangle=posed_xyz,
        rest_xyz_triangle=rest_xyz,
        posed_screen_triangle=posed_screen,
        previous_xyz_triangle=rest_xyz,
        min_projected_double_area_px2=1.0,
    )
    assert metrics["relative_surface_condition_number"] == pytest.approx(1.0)
    assert metrics["relative_surface_principal_stretch"] == pytest.approx(1.0)
    assert metrics["uv_to_screen_condition_number_diagnostic"] == pytest.approx(2.0)
    assert metrics["uv_to_surface_condition_number"] == pytest.approx(1.0)


def test_dynamic_appearance_metrics_expose_anisotropic_art_stretch():
    rest = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    posed = ((0.0, 0.0), (60.0, 0.0), (0.0, 10.0))
    uv = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    metrics = dynamic_face_conditioning_metrics(
        uv_triangle=uv,
        posed_xyz_triangle=((0.0, 0.0, 0.0), (60.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        rest_xyz_triangle=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        posed_screen_triangle=posed,
        previous_xyz_triangle=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        min_projected_double_area_px2=1.0,
    )
    assert metrics["relative_surface_condition_number"] == pytest.approx(6.0)
    assert metrics["relative_surface_principal_stretch"] == pytest.approx(6.0)
    assert metrics["adjacent_frame_surface_principal_stretch"] == pytest.approx(6.0)

    policy = validate_dynamic_appearance_policy(_policy())
    assert (
        metrics["relative_surface_principal_stretch"]
        > policy["dynamic_max_relative_surface_principal_stretch"]
    )
    assert (
        metrics["adjacent_frame_surface_principal_stretch"]
        > policy["dynamic_max_adjacent_frame_surface_principal_stretch"]
    )


def test_dynamic_appearance_policy_is_subject_free_and_nontrivial():
    policy_doc = json.loads(
        (
            ROOT
            / "canonical"
            / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
        ).read_text(encoding="utf-8")
    )
    policy = validate_dynamic_appearance_policy(
        policy_doc["completion_quality_policy"]
    )
    assert policy_doc["status"] == "REOPENED_SUBJECT_FREE_VISUAL_FIDELITY_RECALIBRATION_V2"
    assert "KNIGHT_RESULT" in set(policy_doc["forbidden_inputs"])
    assert 1.0 < policy["dynamic_max_relative_surface_condition_number"] < 20.0
    assert 1.0 < policy["dynamic_max_relative_surface_principal_stretch"] < 10.0
    assert 1.0 < policy["dynamic_max_adjacent_frame_surface_principal_stretch"] < 8.0


def test_subject_free_calibration_bank_selects_frozen_relative_thresholds():
    calibration = json.loads(
        (
            ROOT
            / "canonical"
            / "DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json"
        ).read_text(encoding="utf-8")
    )
    policy = validate_dynamic_appearance_policy(_policy())
    selected = calibration["selected_policy"]
    assert calibration["status"] == "PASS_SUBJECT_FREE_SYNTHETIC_AND_ARTICULATED_LBS_CALIBRATION_V2"
    assert calibration["witness_used"] is False
    assert calibration["threshold_tuning_from_knight_forbidden"] is True
    assert policy["dynamic_max_relative_surface_condition_number"] == selected[
        "dynamic_max_relative_surface_condition_number"
    ]
    assert policy["dynamic_max_relative_surface_principal_stretch"] == selected[
        "dynamic_max_relative_surface_principal_stretch"
    ]
    assert policy["dynamic_max_adjacent_frame_surface_principal_stretch"] == selected[
        "dynamic_max_adjacent_frame_surface_principal_stretch"
    ]

    uv = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    rest_xyz = (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
    )
    for row in calibration["benign_cases"]:
        sx, sy = map(float, row["surface_scale_xy"])
        posed_xyz = (
            (0.0, 0.0, 0.0),
            (10.0 * sx, 0.0, 0.0),
            (0.0, 10.0 * sy, 0.0),
        )
        metrics = dynamic_face_conditioning_metrics(
            uv_triangle=uv,
            posed_xyz_triangle=posed_xyz,
            rest_xyz_triangle=rest_xyz,
            posed_screen_triangle=(
                (0.0, 0.0),
                (10.0 * sx, 0.0),
                (0.0, 10.0 * sy),
            ),
            previous_xyz_triangle=rest_xyz,
            min_projected_double_area_px2=1.0,
        )
        assert metrics["relative_surface_condition_number"] <= policy[
            "dynamic_max_relative_surface_condition_number"
        ]
        assert metrics["relative_surface_principal_stretch"] <= policy[
            "dynamic_max_relative_surface_principal_stretch"
        ]

    for row in calibration["adversarial_cases"]:
        sx, sy = map(float, row["surface_scale_xy"])
        posed_xyz = (
            (0.0, 0.0, 0.0),
            (10.0 * sx, 0.0, 0.0),
            (0.0, 10.0 * sy, 0.0),
        )
        metrics = dynamic_face_conditioning_metrics(
            uv_triangle=uv,
            posed_xyz_triangle=posed_xyz,
            rest_xyz_triangle=rest_xyz,
            posed_screen_triangle=(
                (0.0, 0.0),
                (10.0 * sx, 0.0),
                (0.0, 10.0 * sy),
            ),
            previous_xyz_triangle=rest_xyz,
            min_projected_double_area_px2=1.0,
        )
        assert (
            metrics["relative_surface_condition_number"]
            > policy["dynamic_max_relative_surface_condition_number"]
            or metrics["relative_surface_principal_stretch"]
            > policy["dynamic_max_relative_surface_principal_stretch"]
        )


def _lbs_hinge_pose(angle_degrees: float):
    rest = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.4, 0.0)),
        dtype=np.float64,
    )
    theta = math.radians(float(angle_degrees))
    rotation = np.asarray(
        (
            (math.cos(theta), -math.sin(theta), 0.0),
            (math.sin(theta), math.cos(theta), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    weights = (0.0, 1.0, 0.5)
    posed = np.stack(
        [
            (1.0 - weight) * point + weight * (rotation @ point)
            for point, weight in zip(rest, weights)
        ],
        axis=0,
    )
    return rest, posed


def test_articulated_lbs_bank_selects_three_for_intrinsic_and_two_for_temporal():
    calibration = json.loads(
        (
            ROOT
            / "canonical"
            / "DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json"
        ).read_text(encoding="utf-8")
    )
    policy = validate_dynamic_appearance_policy(_policy())
    uv = np.asarray(calibration["articulated_lbs_contract"]["uv_triangle"], dtype=np.float64)

    benign_max_condition = 1.0
    benign_max_stretch = 1.0
    for row in calibration["articulated_benign_cases"]:
        rest, posed = _lbs_hinge_pose(row["angle_degrees"])
        metrics = dynamic_face_conditioning_metrics(
            uv_triangle=uv,
            posed_xyz_triangle=posed,
            rest_xyz_triangle=rest,
            posed_screen_triangle=posed[:, :2] * 100.0,
            previous_xyz_triangle=rest,
            min_projected_double_area_px2=1.0,
        )
        benign_max_condition = max(
            benign_max_condition,
            metrics["relative_surface_condition_number"],
            metrics["uv_to_surface_condition_number"],
        )
        benign_max_stretch = max(
            benign_max_stretch,
            metrics["relative_surface_principal_stretch"],
        )
        assert metrics["relative_surface_condition_number"] <= policy[
            "dynamic_max_relative_surface_condition_number"
        ]
        assert metrics["relative_surface_principal_stretch"] <= policy[
            "dynamic_max_relative_surface_principal_stretch"
        ]
        assert metrics["uv_to_surface_condition_number"] <= policy[
            "dynamic_max_uv_to_surface_condition_number"
        ]

    adversarial_rejected = []
    for row in calibration["articulated_adversarial_cases"]:
        rest, posed = _lbs_hinge_pose(row["angle_degrees"])
        metrics = dynamic_face_conditioning_metrics(
            uv_triangle=uv,
            posed_xyz_triangle=posed,
            rest_xyz_triangle=rest,
            posed_screen_triangle=posed[:, :2] * 100.0,
            previous_xyz_triangle=rest,
            min_projected_double_area_px2=1.0,
        )
        adversarial_rejected.append(
            metrics["relative_surface_condition_number"]
            > policy["dynamic_max_relative_surface_condition_number"]
            or metrics["relative_surface_principal_stretch"]
            > policy["dynamic_max_relative_surface_principal_stretch"]
            or metrics["uv_to_surface_condition_number"]
            > policy["dynamic_max_uv_to_surface_condition_number"]
        )
    assert all(adversarial_rejected)
    assert benign_max_condition > 2.5
    assert benign_max_stretch > 2.4
    assert policy["dynamic_max_relative_surface_condition_number"] == 3.0
    assert policy["dynamic_max_relative_surface_principal_stretch"] == 3.0
    assert policy["dynamic_max_uv_to_surface_condition_number"] == 3.0

    rest, previous = _lbs_hinge_pose(30.0)
    _rest_again, current = _lbs_hinge_pose(45.0)
    temporal = dynamic_face_conditioning_metrics(
        uv_triangle=uv,
        posed_xyz_triangle=current,
        rest_xyz_triangle=rest,
        posed_screen_triangle=current[:, :2] * 100.0,
        previous_xyz_triangle=previous,
        min_projected_double_area_px2=1.0,
    )
    assert temporal["adjacent_frame_surface_principal_stretch"] < 2.0

    _rest_again, severe = _lbs_hinge_pose(60.0)
    temporal_bad = dynamic_face_conditioning_metrics(
        uv_triangle=uv,
        posed_xyz_triangle=severe,
        rest_xyz_triangle=rest,
        posed_screen_triangle=severe[:, :2] * 100.0,
        previous_xyz_triangle=current,
        min_projected_double_area_px2=1.0,
    )
    assert temporal_bad["adjacent_frame_surface_principal_stretch"] > 1.9
    assert policy["dynamic_max_adjacent_frame_surface_principal_stretch"] == 2.0
