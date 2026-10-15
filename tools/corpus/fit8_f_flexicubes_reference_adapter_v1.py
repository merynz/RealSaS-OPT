from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

SCHEMA_VERSION = "RealSaS.FFlexiCubesReferenceAdapter.v1"
UPSTREAM_REPOSITORY = "nv-tlabs/FlexiCubes"
UPSTREAM_COMMIT_SHA = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"

# RealSaS chart-lattice corner order matches upstream FlexiCubes exactly.
CUBE_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)

# RealSaS chart-lattice edge order.
CHART_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# Exact edge order used by nv-tlabs/FlexiCubes at the bound upstream commit.
FLEXICUBES_EDGES = (
    (0, 1), (1, 5), (4, 5), (0, 4),
    (2, 3), (3, 7), (6, 7), (2, 6),
    (2, 0), (3, 1), (7, 5), (6, 4),
)


def _derive_edge_binding() -> tuple[tuple[int, bool], ...]:
    rows: list[tuple[int, bool]] = []
    used: set[int] = set()
    for chart_a, chart_b in CHART_EDGES:
        match = None
        for flex_index, (flex_a, flex_b) in enumerate(FLEXICUBES_EDGES):
            if (chart_a, chart_b) == (flex_a, flex_b):
                match = (flex_index, False)
                break
            if (chart_a, chart_b) == (flex_b, flex_a):
                match = (flex_index, True)
                break
        if match is None:
            raise RuntimeError("FLEXICUBES_REFERENCE_EDGE_BINDING_INCOMPLETE")
        if match[0] in used:
            raise RuntimeError("FLEXICUBES_REFERENCE_EDGE_BINDING_NOT_BIJECTIVE")
        used.add(match[0])
        rows.append(match)
    if used != set(range(12)):
        raise RuntimeError("FLEXICUBES_REFERENCE_EDGE_BINDING_NOT_PERMUTATION")
    return tuple(rows)


# For each RealSaS chart edge: (upstream FlexiCubes edge index, orientation reversed?).
CHART_TO_FLEXICUBES_EDGE = _derive_edge_binding()


def chart_edge_crossing_to_flexicubes_v1(edge_index: int, t: float) -> tuple[int, float]:
    if edge_index < 0 or edge_index >= len(CHART_EDGES):
        raise ValueError("FLEXICUBES_REFERENCE_CHART_EDGE_OOB")
    if not isfinite(t) or t < 0.0 or t > 1.0:
        raise ValueError("FLEXICUBES_REFERENCE_EDGE_T_INVALID")
    flex_index, reversed_orientation = CHART_TO_FLEXICUBES_EDGE[edge_index]
    return flex_index, (1.0 - t if reversed_orientation else t)


def remap_chart_edge_values_to_flexicubes_v1(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 12:
        raise ValueError("FLEXICUBES_REFERENCE_EDGE_VALUE_ARITY_INVALID")
    out = [0.0] * 12
    for chart_index, value in enumerate(values):
        if not isfinite(float(value)):
            raise ValueError("FLEXICUBES_REFERENCE_EDGE_VALUE_NONFINITE")
        flex_index, _ = CHART_TO_FLEXICUBES_EDGE[chart_index]
        out[flex_index] = float(value)
    return tuple(out)


@dataclass(frozen=True)
class FlexiCubesDenseGridV1:
    cell_min_ijk: tuple[int, int, int]
    cell_max_exclusive_ijk: tuple[int, int, int]
    lattice_origin_xyz: tuple[float, float, float]
    extraction_pitch: float
    lattice_ijks: tuple[tuple[int, int, int], ...]
    vertex_xyz: tuple[tuple[float, float, float], ...]
    cube_fx8: tuple[tuple[int, ...], ...]
    cube_ijks: tuple[tuple[int, int, int], ...]
    schema_version: str = SCHEMA_VERSION

    @property
    def res(self) -> tuple[int, int, int]:
        return tuple(
            self.cell_max_exclusive_ijk[i] - self.cell_min_ijk[i]
            for i in range(3)
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FLEXICUBES_REFERENCE_GRID_SCHEMA_INVALID")
        if len(self.cell_min_ijk) != 3 or len(self.cell_max_exclusive_ijk) != 3:
            raise ValueError("FLEXICUBES_REFERENCE_GRID_BOUNDS_INVALID")
        if any(v <= 0 for v in self.res):
            raise ValueError("FLEXICUBES_REFERENCE_GRID_RES_INVALID")
        if not isfinite(self.extraction_pitch) or self.extraction_pitch <= 0.0:
            raise ValueError("FLEXICUBES_REFERENCE_GRID_PITCH_INVALID")
        if len(self.lattice_origin_xyz) != 3 or any(not isfinite(v) for v in self.lattice_origin_xyz):
            raise ValueError("FLEXICUBES_REFERENCE_GRID_ORIGIN_INVALID")
        expected_vertex_count = (self.res[0] + 1) * (self.res[1] + 1) * (self.res[2] + 1)
        expected_cube_count = self.res[0] * self.res[1] * self.res[2]
        if len(self.lattice_ijks) != expected_vertex_count or len(self.vertex_xyz) != expected_vertex_count:
            raise ValueError("FLEXICUBES_REFERENCE_GRID_VERTEX_COUNT_INVALID")
        if len(self.cube_fx8) != expected_cube_count or len(self.cube_ijks) != expected_cube_count:
            raise ValueError("FLEXICUBES_REFERENCE_GRID_CUBE_COUNT_INVALID")
        if len(set(self.lattice_ijks)) != len(self.lattice_ijks):
            raise ValueError("FLEXICUBES_REFERENCE_GRID_DUPLICATE_VERTEX_IJK")
        if len(set(self.cube_ijks)) != len(self.cube_ijks):
            raise ValueError("FLEXICUBES_REFERENCE_GRID_DUPLICATE_CUBE_IJK")
        for cube in self.cube_fx8:
            if len(cube) != 8:
                raise ValueError("FLEXICUBES_REFERENCE_GRID_CUBE_ARITY_INVALID")
            if min(cube) < 0 or max(cube) >= expected_vertex_count:
                raise ValueError("FLEXICUBES_REFERENCE_GRID_CUBE_VERTEX_OOB")


def build_dense_flexicubes_grid_v1(
    *,
    cell_min_ijk: tuple[int, int, int],
    cell_max_exclusive_ijk: tuple[int, int, int],
    lattice_origin_xyz: tuple[float, float, float],
    extraction_pitch: float,
) -> FlexiCubesDenseGridV1:
    if len(cell_min_ijk) != 3 or len(cell_max_exclusive_ijk) != 3:
        raise ValueError("FLEXICUBES_REFERENCE_BUILD_BOUNDS_INVALID")
    res = tuple(cell_max_exclusive_ijk[i] - cell_min_ijk[i] for i in range(3))
    if any(v <= 0 for v in res):
        raise ValueError("FLEXICUBES_REFERENCE_BUILD_RES_INVALID")
    if not isfinite(extraction_pitch) or extraction_pitch <= 0.0:
        raise ValueError("FLEXICUBES_REFERENCE_BUILD_PITCH_INVALID")

    min_x, min_y, min_z = cell_min_ijk
    rx, ry, rz = res
    lattice_ijks: list[tuple[int, int, int]] = []
    vertex_xyz: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[int, int, int], int] = {}

    # PyTorch nonzero over a dense [X,Y,Z] volume is lexicographic with Z fastest.
    for x in range(min_x, min_x + rx + 1):
        for y in range(min_y, min_y + ry + 1):
            for z in range(min_z, min_z + rz + 1):
                ijk = (x, y, z)
                vertex_index[ijk] = len(lattice_ijks)
                lattice_ijks.append(ijk)
                vertex_xyz.append(tuple(
                    float(lattice_origin_xyz[axis]) + extraction_pitch * float(ijk[axis])
                    for axis in range(3)
                ))

    cube_fx8: list[tuple[int, ...]] = []
    cube_ijks: list[tuple[int, int, int]] = []
    for x in range(min_x, min_x + rx):
        for y in range(min_y, min_y + ry):
            for z in range(min_z, min_z + rz):
                cube_ijk = (x, y, z)
                cube_ijks.append(cube_ijk)
                cube_fx8.append(tuple(
                    vertex_index[(x + dx, y + dy, z + dz)]
                    for dx, dy, dz in CUBE_CORNERS
                ))

    result = FlexiCubesDenseGridV1(
        cell_min_ijk=cell_min_ijk,
        cell_max_exclusive_ijk=cell_max_exclusive_ijk,
        lattice_origin_xyz=tuple(float(v) for v in lattice_origin_xyz),
        extraction_pitch=float(extraction_pitch),
        lattice_ijks=tuple(lattice_ijks),
        vertex_xyz=tuple(vertex_xyz),
        cube_fx8=tuple(cube_fx8),
        cube_ijks=tuple(cube_ijks),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FlexiCubesScalarBindingV1:
    scalar_values: tuple[float, ...]
    orientation_valid: tuple[bool, ...]
    source_binding: tuple[str, ...]
    global_sign_authority: bool
    support_clipping_required: bool
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self, grid: FlexiCubesDenseGridV1) -> None:
        grid.validate()
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FLEXICUBES_REFERENCE_SCALAR_SCHEMA_INVALID")
        count = len(grid.lattice_ijks)
        if len(self.scalar_values) != count or len(self.orientation_valid) != count or len(self.source_binding) != count:
            raise ValueError("FLEXICUBES_REFERENCE_SCALAR_VERTEX_COUNT_MISMATCH")
        if any(not isfinite(value) for value in self.scalar_values):
            raise ValueError("FLEXICUBES_REFERENCE_SCALAR_NONFINITE")
        if any(not value for value in self.source_binding):
            raise ValueError("FLEXICUBES_REFERENCE_SOURCE_BINDING_MISSING")
        if not self.provenance:
            raise ValueError("FLEXICUBES_REFERENCE_SCALAR_PROVENANCE_MISSING")

    @property
    def vanilla_upstream_admitted(self) -> bool:
        return (
            self.global_sign_authority
            and not self.support_clipping_required
            and all(self.orientation_valid)
        )


def validate_vanilla_flexicubes_reference_admission_v1(
    grid: FlexiCubesDenseGridV1,
    binding: FlexiCubesScalarBindingV1,
) -> None:
    binding.validate(grid)
    if not binding.global_sign_authority:
        raise ValueError("FLEXICUBES_REFERENCE_GLOBAL_SIGN_AUTHORITY_REQUIRED")
    if binding.support_clipping_required:
        raise ValueError("FLEXICUBES_REFERENCE_SUPPORT_CLIPPING_UNSUPPORTED")
    if not all(binding.orientation_valid):
        raise ValueError("FLEXICUBES_REFERENCE_ORIENTATION_VALIDITY_REQUIRED")


__all__ = [
    "CHART_EDGES",
    "CHART_TO_FLEXICUBES_EDGE",
    "CUBE_CORNERS",
    "FLEXICUBES_EDGES",
    "FlexiCubesDenseGridV1",
    "FlexiCubesScalarBindingV1",
    "SCHEMA_VERSION",
    "UPSTREAM_COMMIT_SHA",
    "UPSTREAM_REPOSITORY",
    "build_dense_flexicubes_grid_v1",
    "chart_edge_crossing_to_flexicubes_v1",
    "remap_chart_edge_values_to_flexicubes_v1",
    "validate_vanilla_flexicubes_reference_admission_v1",
]
