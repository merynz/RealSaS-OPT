import numpy as np

from tools.corpus.fit8_static_self_intersection_qualification_v1 import (
    StaticSelfIntersectionPolicyV1,
    qualify_static_self_intersection_v1,
)


CUBE_VERTICES = np.asarray([
    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
    [-1.0, -1.0,  1.0], [1.0, -1.0,  1.0], [1.0, 1.0,  1.0], [-1.0, 1.0,  1.0],
], dtype=np.float64)

CUBE_FACES = np.asarray([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
], dtype=np.int64)


def _policy():
    return StaticSelfIntersectionPolicyV1(
        provenance="synthetic_static_intersection_guard",
    )


def test_clean_closed_cube_is_stably_intersection_free():
    result = qualify_static_self_intersection_v1(
        vertices=CUBE_VERTICES,
        faces=CUBE_FACES,
        policy=_policy(),
    )
    assert result.qualified
    assert result.stable_across_tolerance_sweep
    assert result.self_intersection_free
    assert all(len(rows) == 0 for rows in result.pair_sets)
    assert result.maximum_test_tolerance > 0.0
    assert result.tested_tolerances[0] == 0.0
    assert result.tested_tolerances[-1] == result.maximum_test_tolerance


def test_true_nonadjacent_triangle_crossing_is_stably_reported():
    vertices = np.asarray([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.75, 0.0],
    ], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    result = qualify_static_self_intersection_v1(
        vertices=vertices,
        faces=faces,
        policy=_policy(),
    )
    assert result.qualified
    assert result.stable_across_tolerance_sweep
    assert not result.self_intersection_free
    assert result.pair_sets[0] == ((0, 1),)
    assert all(rows == ((0, 1),) for rows in result.pair_sets)


def test_uniform_scale_changes_derived_tolerance_not_intersection_decision():
    base = qualify_static_self_intersection_v1(
        vertices=CUBE_VERTICES,
        faces=CUBE_FACES,
        policy=_policy(),
    )
    scaled = qualify_static_self_intersection_v1(
        vertices=CUBE_VERTICES * 1000.0,
        faces=CUBE_FACES,
        policy=_policy(),
    )
    assert scaled.qualified == base.qualified
    assert scaled.self_intersection_free == base.self_intersection_free
    assert np.isclose(scaled.minimum_triangle_altitude, 1000.0 * base.minimum_triangle_altitude)
    assert np.isclose(scaled.maximum_test_tolerance, 1000.0 * base.maximum_test_tolerance)


def test_invalid_policy_cannot_hide_absolute_magic_tolerance():
    bad = StaticSelfIntersectionPolicyV1(
        tolerance_fraction_of_shrink_clearance=0.75,
        provenance="bad_guard",
    )
    try:
        bad.validate()
    except ValueError as exc:
        assert "CLEARANCE_FRACTION_INVALID" in str(exc)
    else:
        raise AssertionError("expected fail-closed tolerance policy")
