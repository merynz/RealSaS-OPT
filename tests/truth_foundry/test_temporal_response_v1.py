from __future__ import annotations

import inspect

import numpy as np
import pytest

from models.iris.truth.temporal_response_v1 import (
    DYNAMIC_AUTHORITY,
    RESPONSE_SEMANTICS,
    build_temporal_response_candidate_v1,
)


def _fixture():
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    posed = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.2, 0.0], [1.0, 0.4, 0.0]],
        ],
        dtype=np.float64,
    )
    rows = [
        {"sample_index": 0, "action": "walk", "time_seconds": 0.0},
        {"sample_index": 1, "action": "walk", "time_seconds": 0.5},
        {"sample_index": 2, "action": "walk", "time_seconds": 1.0},
        {"sample_index": 3, "action": "idle", "time_seconds": 0.0},
        {"sample_index": 4, "action": "idle", "time_seconds": 1.0},
    ]
    return rest, posed, rows


def test_one_authored_action_becomes_locus_condition_time_vector_response():
    rest, posed, rows = _fixture()
    result = build_temporal_response_candidate_v1(rest, posed, rows, action_identity="walk")
    assert result.response.shape == (2, 1, 3, 3)
    assert result.timestamps_seconds.tolist() == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(result.response[:, 0, 0], np.zeros((2, 3)))
    np.testing.assert_allclose(result.response[0, 0, 2], [0.3, 0.0, 0.0])
    np.testing.assert_allclose(result.response[1, 0, 2], [0.5, 0.0, 0.0])
    assert result.sample_validity.shape == (2, 1, 3)
    assert result.interval_validity.shape == (2, 1, 2)
    assert result.sample_validity.all()
    assert result.interval_validity.all()


def test_authored_actions_never_share_one_time_axis():
    rest, posed, rows = _fixture()
    walk = build_temporal_response_candidate_v1(rest, posed, rows, action_identity="walk")
    idle = build_temporal_response_candidate_v1(rest, posed, rows, action_identity="idle")
    assert walk.frame_count == 3
    assert idle.frame_count == 2
    assert idle.timestamps_seconds.tolist() == [0.0, 1.0]
    np.testing.assert_allclose(idle.response[0, 0, 1], [0.0, 0.2, 0.0])


def test_source_world_root_motion_is_preserved_not_gauged_away():
    rest, posed, rows = _fixture()
    result = build_temporal_response_candidate_v1(rest, posed, rows, action_identity="walk")
    assert result.response_semantics == RESPONSE_SEMANTICS
    # Vertex zero carries the authored global translation exactly.
    np.testing.assert_allclose(result.response[0, 0, 1], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(result.response[0, 0, 2], [0.3, 0.0, 0.0])


def test_measured_authored_clip_does_not_mint_physics_or_product_authority():
    rest, posed, rows = _fixture()
    result = build_temporal_response_candidate_v1(rest, posed, rows, action_identity="walk")
    assert result.dynamic_authority == DYNAMIC_AUTHORITY
    assert result.qualification_minted is False
    assert result.student_target_allowed is False
    assert result.compiler_consequence_allowed is False
    assert "PHYSICS" in result.dynamic_authority


def test_nonzero_local_time_origin_and_nonmonotonic_time_fail_closed():
    rest, posed, rows = _fixture()
    bad = [dict(row) for row in rows]
    bad[0]["time_seconds"] = 0.25
    with pytest.raises(ValueError, match="LOCAL_ZERO"):
        build_temporal_response_candidate_v1(rest, posed, bad, action_identity="walk")

    bad = [dict(row) for row in rows]
    bad[2]["time_seconds"] = 0.5
    with pytest.raises(ValueError, match="STRICTLY_INCREASING"):
        build_temporal_response_candidate_v1(rest, posed, bad, action_identity="walk")


def test_kernel_api_has_no_physical_dof_impulse_or_semantic_part_inputs():
    parameters = set(inspect.signature(build_temporal_response_candidate_v1).parameters)
    forbidden = {
        "joint_limit",
        "physical_dof",
        "impulse",
        "force",
        "cape",
        "hair",
        "secondary_motion_class",
        "semantic_part",
        "bone_name",
    }
    assert parameters.isdisjoint(forbidden)
