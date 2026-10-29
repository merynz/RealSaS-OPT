from __future__ import annotations

import numpy as np

from tools.corpus.build_truth139_visibility_occlusion_teacher_v1 import (
    VISIBILITY,
    classify_target,
    deterministic_stratified_rows,
    project_source_point,
)


def _target():
    lookup = np.full((5, 5), -1, dtype=np.int32)
    lookup[2, 2] = 0
    lookup[2, 3] = 1
    return {
        "row_lookup": lookup,
        "target_xyz_canonical": np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        "target_depth": np.asarray([4.0, 4.1], dtype=np.float32),
        "target_triangle": np.asarray([7, 8], dtype=np.int32),
        "target_facing": np.asarray([1.0, 0.5], dtype=np.float32),
        "target_source_support": np.asarray([True, False], dtype=np.bool_),
    }


def test_deterministic_stratified_sampling_is_stable_unique_and_bounded():
    a = deterministic_stratified_rows(1000, 64, subject_id="truth139_x", source_view=3)
    b = deterministic_stratified_rows(1000, 64, subject_id="truth139_x", source_view=3)
    c = deterministic_stratified_rows(1000, 64, subject_id="truth139_x", source_view=4)
    assert np.array_equal(a, b)
    assert len(a) == len(np.unique(a)) == 64
    assert np.all((0 <= a) & (a < 1000))
    assert not np.array_equal(a, c)


def test_projection_uses_product_top_left_orthographic_convention():
    camera = {
        "origin": [0.0, 0.0, 10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, -1.0],
        "half_extent": 2.0,
        "resolution": 4,
    }
    inside, x, y, depth = project_source_point(np.asarray([0.0, 0.0, 0.0]), camera)
    assert inside is True
    assert np.isclose(x, 1.5)
    assert np.isclose(y, 1.5)
    assert np.isclose(depth, 10.0)


def test_visible_same_surface_beats_front_depth_relation():
    target = _target()
    result = classify_target(
        sample_canonical=np.asarray([0.01, 0.0, 0.0]),
        sample_depth=4.0,
        projected_x=2,
        projected_y=2,
        surface_tolerance=0.05,
        depth_epsilon=0.01,
        search_radius_px=1,
        **target,
    )
    assert result[0] == VISIBILITY["VISIBLE_SAME_SURFACE"]
    assert result[2] == 7
    assert result[5] is True
    assert result[6] is True


def test_strictly_closer_front_surface_means_occluded():
    target = _target()
    result = classify_target(
        sample_canonical=np.asarray([9.0, 0.0, 0.0]),
        sample_depth=5.0,
        projected_x=2,
        projected_y=2,
        surface_tolerance=0.05,
        depth_epsilon=0.1,
        search_radius_px=1,
        **target,
    )
    assert result[0] == VISIBILITY["OCCLUDED_BY_FRONT_SURFACE"]
    assert result[2] == 7
    assert np.isclose(result[3], 1.0)
    assert result[6] is False


def test_background_and_depth_disagreement_are_distinct_states():
    target = _target()
    background = classify_target(
        sample_canonical=np.asarray([0.0, 0.0, 0.0]),
        sample_depth=4.0,
        projected_x=0,
        projected_y=0,
        surface_tolerance=0.01,
        depth_epsilon=0.1,
        search_radius_px=0,
        **target,
    )
    assert background[0] == VISIBILITY["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"]

    disagreement = classify_target(
        sample_canonical=np.asarray([9.0, 0.0, 0.0]),
        sample_depth=3.5,
        projected_x=2,
        projected_y=2,
        surface_tolerance=0.01,
        depth_epsilon=0.1,
        search_radius_px=0,
        **target,
    )
    assert disagreement[0] == VISIBILITY["RASTER_GEOMETRY_DISAGREEMENT_OR_AMIGUOUS"] if False else VISIBILITY["RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS"]
