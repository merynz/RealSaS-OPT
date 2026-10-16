from __future__ import annotations

import math

import numpy as np
import pytest

from models.iris.vnext.teacher_truth_v1 import build_mechanical_demand_truth_v1


def test_rank_deficient_authored_poses_remain_valid_mechanical_evidence():
    rest = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    stretched = np.array(
        [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    line_collapse = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    point_collapse = np.array(
        [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]],
        dtype=np.float64,
    )

    truth = build_mechanical_demand_truth_v1(
        rest,
        faces,
        np.stack((rest, stretched, line_collapse, point_collapse), axis=0),
    )

    assert truth.frame_count == 4
    assert truth.metadata["pose_degenerate_face_samples"] == 2
    assert truth.metadata["pose_degenerate_face_count"] == 1
    assert truth.metadata["pose_degenerate_frame_count"] == 2
    assert truth.metadata["pose_rank_zero_face_samples"] == 1
    assert truth.metadata["posed_rank_deficiency_policy"].startswith(
        "VALID_DEFORMATION_EVIDENCE__3X2_DEFORMATION_GRADIENT"
    )
    assert truth.area_change_p95[0] == pytest.approx(1.0)
    assert truth.bend_demand[0] == pytest.approx(math.pi)
    assert truth.rigidity_probability[0] < 1.0
    assert np.isfinite(np.asarray(truth.condition_number_p95)).all()


def test_regular_pose_uses_expected_material_deformation_singular_values():
    rest = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    posed = np.array(
        [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [0.0, 0.8, 0.0]],
        dtype=np.float64,
    )

    truth = build_mechanical_demand_truth_v1(rest, faces, posed[None])

    assert truth.stretch_major[0] == pytest.approx(0.25)
    assert truth.stretch_minor[0] == pytest.approx(0.20)
    assert truth.area_change_p95[0] == pytest.approx(0.0)
    assert truth.tangent_direction_2theta[0] == pytest.approx((1.0, 0.0))


def test_rest_topology_still_fails_closed_when_degenerate():
    rest = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="MECHANICAL_TRUTH_DEGENERATE_FACE"):
        build_mechanical_demand_truth_v1(rest, faces, rest[None])
