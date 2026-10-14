from tools.corpus.fit8_cube_rotation_symmetry_v1 import (
    IDENTITY_MATRIX,
    _det3,
    _mat_mul,
    compose_cube_rotations_v1,
    proper_cube_rotations_v1,
    reflection_matrix_example_v1,
    rotation_by_matrix_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import CUBE_EDGE_CORNERS
from tools.corpus.fit8_patch_face_support_v1 import CELL_FACE_AXIS_SIDE


def test_exactly_twenty_four_proper_cube_rotations_exist():
    rotations = proper_cube_rotations_v1()
    assert len(rotations) == 24
    assert len({rotation.matrix for rotation in rotations}) == 24
    assert all(_det3(rotation.matrix) == 1 for rotation in rotations)


def test_identity_rotation_is_present():
    identity = rotation_by_matrix_v1(IDENTITY_MATRIX)
    assert identity.corner_permutation == tuple(range(8))
    assert identity.edge_permutation == tuple(range(12))
    assert identity.face_permutation == tuple(range(6))


def test_every_rotation_is_bijective_on_corners_edges_and_faces():
    for rotation in proper_cube_rotations_v1():
        rotation.validate()
        assert sorted(rotation.corner_permutation) == list(range(8))
        assert sorted(rotation.edge_permutation) == list(range(12))
        assert sorted(rotation.face_permutation) == list(range(6))


def test_edge_permutation_preserves_corner_incidence():
    for rotation in proper_cube_rotations_v1():
        for edge_index, (a, b) in enumerate(CUBE_EDGE_CORNERS):
            mapped_edge = CUBE_EDGE_CORNERS[rotation.edge_permutation[edge_index]]
            mapped_corners = {
                rotation.corner_permutation[int(a)],
                rotation.corner_permutation[int(b)],
            }
            assert mapped_corners == set(map(int, mapped_edge))


def test_face_permutation_preserves_rotated_axis_normal():
    for rotation in proper_cube_rotations_v1():
        for face_index, (axis, side) in enumerate(CELL_FACE_AXIS_SIDE):
            normal = [0, 0, 0]
            normal[int(axis)] = int(side)
            rotated = tuple(
                sum(rotation.matrix[row][col] * normal[col] for col in range(3))
                for row in range(3)
            )
            mapped_axis, mapped_side = CELL_FACE_AXIS_SIDE[rotation.face_permutation[face_index]]
            expected = [0, 0, 0]
            expected[int(mapped_axis)] = int(mapped_side)
            assert rotated == tuple(expected)


def test_proper_rotations_are_closed_under_composition():
    rotations = proper_cube_rotations_v1()
    matrix_set = {rotation.matrix for rotation in rotations}
    for a in rotations:
        for b in rotations:
            composed = compose_cube_rotations_v1(a, b)
            assert composed.matrix == _mat_mul(a.matrix, b.matrix)
            assert composed.matrix in matrix_set


def test_reflection_is_deliberately_not_a_rotation():
    reflection = reflection_matrix_example_v1()
    assert _det3(reflection) == -1
    assert reflection not in {rotation.matrix for rotation in proper_cube_rotations_v1()}
    try:
        rotation_by_matrix_v1(reflection)
    except ValueError as exc:
        assert "NOT_IN_PROPER_GROUP" in str(exc)
    else:
        raise AssertionError("reflection must not be admitted as a proper cube rotation")
