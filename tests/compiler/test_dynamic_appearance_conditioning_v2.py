from __future__ import annotations

import json
from pathlib import Path

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
        <= policy["dynamic_max_relative_surface_principal_stretch"]
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
    assert policy_doc["status"].startswith("FROZEN_SUBJECT_FREE")
    assert "KNIGHT_RESULT" in set(policy_doc["forbidden_inputs"])
    assert 1.0 < policy["dynamic_max_relative_surface_condition_number"] < 20.0
    assert 1.0 < policy["dynamic_max_relative_surface_principal_stretch"] < 10.0
    assert 1.0 < policy["dynamic_max_adjacent_frame_surface_principal_stretch"] < 8.0
