import numpy as np

from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import (
    triangles_intersect_sat,
    unexpected_intersection_pairs,
)
from tools.corpus.fit8_static_self_intersection_qualification_v1 import (
    StaticSelfIntersectionPolicyV1,
)
from tools.corpus.fit8_static_self_intersection_qualification_v2 import (
    _sat_intersects_batch,
    _unexpected_intersection_pairs_v2,
    qualify_static_self_intersection_v2,
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
        provenance="synthetic_vectorized_static_intersection_guard",
    )


def test_vectorized_sat_matches_compiler_scalar_sat_on_random_nondegenerate_pairs():
    rng = np.random.default_rng(20261015)
    a_rows = []
    b_rows = []
    while len(a_rows) < 64:
        a = rng.normal(size=(3, 3))
        b = rng.normal(size=(3, 3))
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) <= 1.0e-5:
            continue
        if np.linalg.norm(np.cross(b[1] - b[0], b[2] - b[0])) <= 1.0e-5:
            continue
        a_rows.append(a)
        b_rows.append(b)
    a = np.stack(a_rows)
    b = np.stack(b_rows)
    for tolerance in (0.0, 1.0e-12, 1.0e-9, 1.0e-6):
        batch = _sat_intersects_batch(a, b, tolerance=tolerance)
        scalar = np.asarray([
            triangles_intersect_sat(x, y, tolerance=tolerance)
            for x, y in zip(a, b)
        ], dtype=bool)
        assert np.array_equal(batch, scalar)


def test_vectorized_unexpected_pair_census_matches_compiler_on_cube():
    for tolerance in (0.0, 1.0e-12, 1.0e-9):
        expected = unexpected_intersection_pairs(
            vertices=CUBE_VERTICES,
            faces=CUBE_FACES,
            tolerance=tolerance,
            shared_contact_exclusion_fraction=1.0e-6,
        )
        actual = _unexpected_intersection_pairs_v2(
            vertices=CUBE_VERTICES,
            faces=CUBE_FACES,
            tolerance=tolerance,
            shared_contact_exclusion_fraction=1.0e-6,
            block_size=4,
        )
        assert actual == expected == ()


def test_vectorized_unexpected_pair_census_matches_compiler_on_true_crossing():
    vertices = np.asarray([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.75, 0.0],
    ], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    expected = unexpected_intersection_pairs(
        vertices=vertices,
        faces=faces,
        tolerance=0.0,
        shared_contact_exclusion_fraction=1.0e-6,
    )
    actual = _unexpected_intersection_pairs_v2(
        vertices=vertices,
        faces=faces,
        tolerance=0.0,
        shared_contact_exclusion_fraction=1.0e-6,
        block_size=1,
    )
    assert actual == expected == ((0, 1),)


def test_v2_scale_aware_court_keeps_cube_stably_clean():
    result = qualify_static_self_intersection_v2(
        vertices=CUBE_VERTICES,
        faces=CUBE_FACES,
        policy=_policy(),
        block_size=4,
    )
    assert result.qualified
    assert result.stable_across_tolerance_sweep
    assert result.self_intersection_free
    assert all(rows == () for rows in result.pair_sets)
    assert "VECTORIZED_SAT_EQUIVALENT" in result.provenance
