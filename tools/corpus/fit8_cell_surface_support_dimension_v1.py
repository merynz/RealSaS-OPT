from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1

SCHEMA_VERSION = "RealSaS.CellSurfaceSupportDimension.v1"
SUPPORT_NONE = "NONE"
SUPPORT_POINT = "POINT_TOUCH"
SUPPORT_LINE = "LINE_SUPPORT"
SUPPORT_AREA = "AREA_SUPPORT"
SUPPORT_KINDS = {SUPPORT_NONE, SUPPORT_POINT, SUPPORT_LINE, SUPPORT_AREA}


@dataclass(frozen=True)
class SurfaceSupportDimensionPolicyV1:
    distance_tolerance: float
    provenance: str

    def validate(self) -> None:
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0.0:
            raise ValueError("CELL_SURFACE_SUPPORT_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("CELL_SURFACE_SUPPORT_PROVENANCE_MISSING")


@dataclass(frozen=True)
class TriangleCellSupportV1:
    triangle_id: int
    support_kind: str
    clipped_points_xyz: tuple[tuple[float, float, float], ...]
    affine_singular_values: tuple[float, float, float]

    def validate(self) -> None:
        if self.triangle_id < 0:
            raise ValueError("CELL_SURFACE_SUPPORT_TRIANGLE_ID_INVALID")
        if self.support_kind not in SUPPORT_KINDS:
            raise ValueError("CELL_SURFACE_SUPPORT_KIND_INVALID")
        if len(self.affine_singular_values) != 3:
            raise ValueError("CELL_SURFACE_SUPPORT_SINGULAR_VALUES_INVALID")
        if any((not np.isfinite(value)) or value < 0.0 for value in self.affine_singular_values):
            raise ValueError("CELL_SURFACE_SUPPORT_SINGULAR_VALUES_INVALID")
        for point in self.clipped_points_xyz:
            value = np.asarray(point, dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("CELL_SURFACE_SUPPORT_POINT_INVALID")
        if self.support_kind == SUPPORT_NONE and self.clipped_points_xyz:
            raise ValueError("CELL_SURFACE_SUPPORT_NONE_HAS_POINTS")
        if self.support_kind != SUPPORT_NONE and not self.clipped_points_xyz:
            raise ValueError("CELL_SURFACE_SUPPORT_NONEMPTY_MISSING_POINTS")


@dataclass(frozen=True)
class PatchCellSupportV1:
    support_kind: str
    triangle_support: tuple[TriangleCellSupportV1, ...]

    def validate(self) -> None:
        if self.support_kind not in SUPPORT_KINDS:
            raise ValueError("CELL_SURFACE_PATCH_SUPPORT_KIND_INVALID")
        if not self.triangle_support:
            raise ValueError("CELL_SURFACE_PATCH_SUPPORT_EMPTY")
        for row in self.triangle_support:
            row.validate()
        ranks = {SUPPORT_NONE: 0, SUPPORT_POINT: 1, SUPPORT_LINE: 2, SUPPORT_AREA: 3}
        expected = max(self.triangle_support, key=lambda row: ranks[row.support_kind]).support_kind
        if expected != self.support_kind:
            raise ValueError("CELL_SURFACE_PATCH_SUPPORT_KIND_MISMATCH")


def _dedup_points(points: Sequence[np.ndarray], tolerance: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if not any(float(np.linalg.norm(point - prior)) <= tolerance for prior in unique):
            unique.append(point)
    return unique


def _clip_polygon_plane(
    polygon: Sequence[np.ndarray],
    axis: int,
    value: float,
    keep_greater: bool,
    tolerance: float,
) -> list[np.ndarray]:
    if not polygon:
        return []
    out: list[np.ndarray] = []

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return float(point[axis]) >= value - tolerance
        return float(point[axis]) <= value + tolerance

    for index in range(len(polygon)):
        a = np.asarray(polygon[index - 1], dtype=np.float64)
        b = np.asarray(polygon[index], dtype=np.float64)
        ia, ib = inside(a), inside(b)
        if ib:
            if not ia:
                denominator = float(b[axis] - a[axis])
                t = 0.0 if abs(denominator) <= tolerance else float((value - a[axis]) / denominator)
                t = min(1.0, max(0.0, t))
                out.append(a + t * (b - a))
            out.append(b)
        elif ia:
            denominator = float(b[axis] - a[axis])
            t = 0.0 if abs(denominator) <= tolerance else float((value - a[axis]) / denominator)
            t = min(1.0, max(0.0, t))
            out.append(a + t * (b - a))
    return _dedup_points(out, tolerance)


def _clip_triangle_to_cell(
    triangle_xyz: np.ndarray,
    cell: ExtractionCellV1,
    tolerance: float,
) -> list[np.ndarray]:
    center = np.asarray(cell.center_xyz, dtype=np.float64)
    lo = center - float(cell.half_extent)
    hi = center + float(cell.half_extent)
    polygon = [np.asarray(point, dtype=np.float64) for point in triangle_xyz]
    for axis in range(3):
        polygon = _clip_polygon_plane(polygon, axis, float(lo[axis]), True, tolerance)
        if not polygon:
            return []
        polygon = _clip_polygon_plane(polygon, axis, float(hi[axis]), False, tolerance)
        if not polygon:
            return []
    return _dedup_points(polygon, tolerance)


def _affine_singular_values(points: Sequence[np.ndarray]) -> tuple[float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0)
    matrix = np.asarray(points, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    values = np.linalg.svd(centered, compute_uv=False)
    padded = list(map(float, values)) + [0.0, 0.0, 0.0]
    return (padded[0], padded[1], padded[2])


def qualify_triangle_cell_support_v1(
    cell: ExtractionCellV1,
    triangle: SourceTriangleV1,
    *,
    policy: SurfaceSupportDimensionPolicyV1,
) -> TriangleCellSupportV1:
    """Classify authored triangle/cell support by intrinsic geometric dimension.

    The result deliberately separates positive-area surface support from lower-
    dimensional contact. A POINT/LINE contact remains provenance evidence but is
    not sufficient by itself to instantiate an F surface patch.

    Classification uses affine rank under the explicit spatial tolerance rather
    than an implicit area epsilon tied to one extraction resolution.
    """
    cell.validate()
    policy.validate()
    triangle_xyz = triangle.vertices()
    points = _clip_triangle_to_cell(triangle_xyz, cell, float(policy.distance_tolerance))
    singular = _affine_singular_values(points)
    tolerance = float(policy.distance_tolerance)
    if not points:
        kind = SUPPORT_NONE
    elif singular[0] <= tolerance:
        kind = SUPPORT_POINT
    elif singular[1] <= tolerance:
        kind = SUPPORT_LINE
    else:
        kind = SUPPORT_AREA
    result = TriangleCellSupportV1(
        triangle_id=int(triangle.triangle_id),
        support_kind=kind,
        clipped_points_xyz=tuple(tuple(map(float, point)) for point in points),
        affine_singular_values=singular,
    )
    result.validate()
    return result


def qualify_patch_cell_support_v1(
    cell: ExtractionCellV1,
    triangles: Sequence[SourceTriangleV1],
    *,
    policy: SurfaceSupportDimensionPolicyV1,
) -> PatchCellSupportV1:
    if not triangles:
        raise ValueError("CELL_SURFACE_PATCH_SUPPORT_NO_TRIANGLES")
    rows = tuple(
        qualify_triangle_cell_support_v1(cell, triangle, policy=policy)
        for triangle in triangles
    )
    ranks = {SUPPORT_NONE: 0, SUPPORT_POINT: 1, SUPPORT_LINE: 2, SUPPORT_AREA: 3}
    kind = max(rows, key=lambda row: ranks[row.support_kind]).support_kind
    result = PatchCellSupportV1(kind, rows)
    result.validate()
    return result
