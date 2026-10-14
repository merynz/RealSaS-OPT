from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

from tools.corpus.fit8_local_patch_teacher_v1 import CUBE_CORNER_SIGNS, CUBE_EDGE_CORNERS
from tools.corpus.fit8_patch_face_support_v1 import CELL_FACE_AXIS_SIDE

SCHEMA_VERSION = "RealSaS.CubeRotationSymmetry.v1"

Matrix3I = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]


def _det3(m: Matrix3I) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _mat_vec(m: Matrix3I, v: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sum(m[row][col] * v[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _mat_mul(a: Matrix3I, b: Matrix3I) -> Matrix3I:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _transpose(m: Matrix3I) -> Matrix3I:
    return tuple(tuple(m[col][row] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _signed_permutation_matrix(axis_permutation: tuple[int, int, int], signs: tuple[int, int, int]) -> Matrix3I:
    rows = []
    for output_axis in range(3):
        row = [0, 0, 0]
        row[axis_permutation[output_axis]] = signs[output_axis]
        rows.append(tuple(row))
    return tuple(rows)  # type: ignore[return-value]


CORNER_LOOKUP = {tuple(map(int, sign)): index for index, sign in enumerate(CUBE_CORNER_SIGNS)}
EDGE_LOOKUP = {tuple(sorted(map(int, edge))): index for index, edge in enumerate(CUBE_EDGE_CORNERS)}
FACE_LOOKUP = {tuple(map(int, face)): index for index, face in enumerate(CELL_FACE_AXIS_SIDE)}


@dataclass(frozen=True, order=True)
class CubeRotationV1:
    matrix: Matrix3I
    corner_permutation: tuple[int, ...]
    edge_permutation: tuple[int, ...]
    face_permutation: tuple[int, ...]

    def validate(self) -> None:
        if _det3(self.matrix) != 1:
            raise ValueError("CUBE_ROTATION_NOT_PROPER")
        if _mat_mul(self.matrix, _transpose(self.matrix)) != IDENTITY_MATRIX:
            raise ValueError("CUBE_ROTATION_NOT_ORTHONORMAL")
        if tuple(sorted(self.corner_permutation)) != tuple(range(8)):
            raise ValueError("CUBE_ROTATION_CORNER_PERMUTATION_INVALID")
        if tuple(sorted(self.edge_permutation)) != tuple(range(12)):
            raise ValueError("CUBE_ROTATION_EDGE_PERMUTATION_INVALID")
        if tuple(sorted(self.face_permutation)) != tuple(range(6)):
            raise ValueError("CUBE_ROTATION_FACE_PERMUTATION_INVALID")


IDENTITY_MATRIX: Matrix3I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _derive_rotation(matrix: Matrix3I) -> CubeRotationV1:
    corner_permutation = tuple(
        CORNER_LOOKUP[_mat_vec(matrix, tuple(map(int, sign)))]
        for sign in CUBE_CORNER_SIGNS
    )

    edge_permutation = []
    for a, b in CUBE_EDGE_CORNERS:
        rotated_key = tuple(sorted((corner_permutation[int(a)], corner_permutation[int(b)])))
        edge_permutation.append(EDGE_LOOKUP[rotated_key])

    face_permutation = []
    for axis, side in CELL_FACE_AXIS_SIDE:
        normal = [0, 0, 0]
        normal[int(axis)] = int(side)
        rotated = _mat_vec(matrix, tuple(normal))
        nonzero = [index for index, value in enumerate(rotated) if value]
        if len(nonzero) != 1 or abs(rotated[nonzero[0]]) != 1:
            raise ValueError("CUBE_ROTATION_FACE_NORMAL_INVALID")
        rotated_axis = nonzero[0]
        rotated_side = rotated[rotated_axis]
        face_permutation.append(FACE_LOOKUP[(rotated_axis, rotated_side)])

    rotation = CubeRotationV1(
        matrix=matrix,
        corner_permutation=corner_permutation,
        edge_permutation=tuple(edge_permutation),
        face_permutation=tuple(face_permutation),
    )
    rotation.validate()
    return rotation


@lru_cache(maxsize=1)
def proper_cube_rotations_v1() -> tuple[CubeRotationV1, ...]:
    """Return the 24 orientation-preserving symmetries of the cube.

    Reflections are intentionally excluded. RealSaS may later prove that some
    topology signatures are reflection-equivalent, but handedness, side identity,
    normals, appearance and mechanical semantics are not erased by default.
    """
    matrices: set[Matrix3I] = set()
    for axis_permutation_raw in permutations(range(3)):
        axis_permutation = tuple(map(int, axis_permutation_raw))
        for signs_raw in product((-1, 1), repeat=3):
            signs = tuple(map(int, signs_raw))
            matrix = _signed_permutation_matrix(axis_permutation, signs)
            if _det3(matrix) == 1:
                matrices.add(matrix)
    rotations = tuple(_derive_rotation(matrix) for matrix in sorted(matrices))
    if len(rotations) != 24:
        raise RuntimeError(f"CUBE_ROTATION_GROUP_CARDINALITY_INVALID:{len(rotations)}")
    return rotations


def rotation_by_matrix_v1(matrix: Matrix3I) -> CubeRotationV1:
    for rotation in proper_cube_rotations_v1():
        if rotation.matrix == matrix:
            return rotation
    raise ValueError("CUBE_ROTATION_MATRIX_NOT_IN_PROPER_GROUP")


def compose_cube_rotations_v1(a: CubeRotationV1, b: CubeRotationV1) -> CubeRotationV1:
    """Return a after b: matrix(a) @ matrix(b)."""
    a.validate()
    b.validate()
    return rotation_by_matrix_v1(_mat_mul(a.matrix, b.matrix))


def reflection_matrix_example_v1() -> Matrix3I:
    """A determinant -1 mirror used only to assert reflection exclusion."""
    return ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
