from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1, SourceTriangleV1

SCHEMA_VERSION = "RealSaS.FIT8PatchFaceSupport.v1"
# x-, x+, y-, y+, z-, z+
CELL_FACE_AXIS_SIDE = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))


@dataclass(frozen=True)
class PatchFaceSupportRowV1:
    patch_local_id: int
    cell_face_indices: tuple[int, ...]


@dataclass(frozen=True)
class PatchFaceSupportReportV1:
    schema_version: str
    rows: tuple[PatchFaceSupportRowV1, ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FIT8_PATCH_FACE_SUPPORT_SCHEMA_INVALID")
        ids = [row.patch_local_id for row in self.rows]
        if ids != list(range(len(ids))):
            raise ValueError("FIT8_PATCH_FACE_SUPPORT_PATCH_IDS_NOT_CONTIGUOUS")
        for row in self.rows:
            if tuple(sorted(set(row.cell_face_indices))) != row.cell_face_indices:
                raise ValueError("FIT8_PATCH_FACE_SUPPORT_FACE_IDS_NOT_CANONICAL")
            if any(face < 0 or face >= 6 for face in row.cell_face_indices):
                raise ValueError("FIT8_PATCH_FACE_SUPPORT_FACE_ID_INVALID")

    def faces_for_patch(self, patch_local_id: int) -> tuple[int, ...]:
        return self.rows[patch_local_id].cell_face_indices


def _point_in_rect(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> bool:
    return bool(np.all(point >= lo - eps) and np.all(point <= hi + eps))


def _segment_rect_intersects(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> bool:
    # Liang-Barsky clipping in the face's 2D coordinate system.
    t0, t1 = 0.0, 1.0
    delta = p1 - p0
    for axis in range(2):
        if abs(float(delta[axis])) <= eps:
            if p0[axis] < lo[axis] - eps or p0[axis] > hi[axis] + eps:
                return False
            continue
        a = float((lo[axis] - p0[axis]) / delta[axis])
        b = float((hi[axis] - p0[axis]) / delta[axis])
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
        if t0 > t1 + eps:
            return False
    return True


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    a, b, c = triangle
    v0, v1, v2 = c - a, b - a, point - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) <= eps:
        return False
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
    return u >= -eps and v >= -eps and u + v <= 1.0 + eps


def _coplanar_triangle_rect_intersects(triangle: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> bool:
    if any(_point_in_rect(point, lo, hi, eps) for point in triangle):
        return True
    corners = np.asarray(
        ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])),
        dtype=np.float64,
    )
    if any(_point_in_triangle_2d(point, triangle, eps) for point in corners):
        return True
    return any(
        _segment_rect_intersects(triangle[i], triangle[(i + 1) % 3], lo, hi, eps)
        for i in range(3)
    )


def triangle_touches_cell_face_v1(
    triangle_xyz: np.ndarray,
    center_xyz: np.ndarray,
    half_extent: float,
    face_index: int,
    eps: float = 1e-9,
) -> bool:
    triangle = np.asarray(triangle_xyz, dtype=np.float64)
    center = np.asarray(center_xyz, dtype=np.float64)
    if triangle.shape != (3, 3) or center.shape != (3,):
        raise ValueError("FIT8_PATCH_FACE_SUPPORT_GEOMETRY_INVALID")
    if face_index < 0 or face_index >= 6:
        raise ValueError("FIT8_PATCH_FACE_SUPPORT_FACE_ID_INVALID")
    if eps <= 0.0:
        raise ValueError("FIT8_PATCH_FACE_SUPPORT_TOLERANCE_INVALID")
    axis, side = CELL_FACE_AXIS_SIDE[face_index]
    plane = float(center[axis] + side * half_extent)
    distance = triangle[:, axis] - plane
    if np.all(distance > eps) or np.all(distance < -eps):
        return False
    uv_axes = [i for i in range(3) if i != axis]
    lo = center[uv_axes] - half_extent
    hi = center[uv_axes] + half_extent
    if np.all(np.abs(distance) <= eps):
        return _coplanar_triangle_rect_intersects(triangle[:, uv_axes], lo, hi, eps)

    points: list[np.ndarray] = []
    for i in range(3):
        if abs(float(distance[i])) <= eps:
            points.append(triangle[i, uv_axes])
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = float(distance[i]), float(distance[j])
        if di * dj < -(eps * eps):
            t = di / (di - dj)
            points.append((triangle[i] + t * (triangle[j] - triangle[i]))[uv_axes])

    unique: list[np.ndarray] = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if not any(float(np.linalg.norm(point - prior)) <= eps for prior in unique):
            unique.append(point)
    if any(_point_in_rect(point, lo, hi, eps) for point in unique):
        return True
    return any(
        _segment_rect_intersects(unique[i], unique[j], lo, hi, eps)
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
    )


def derive_patch_face_support_v1(
    cell_teacher: ExtractionCellTeacherV1,
    triangles: Sequence[SourceTriangleV1],
    *,
    geometry_tolerance: float = 1e-9,
) -> PatchFaceSupportReportV1:
    """Derive exact teacher evidence for which cube faces each local patch reaches.

    This evidence is needed before treating an unlinked patch in a neighboring-cell
    continuation court as meaningful. Interior patches are not expected to cross a
    particular shared face and must never be counted as stitching failures.
    """
    if geometry_tolerance <= 0.0:
        raise ValueError("FIT8_PATCH_FACE_SUPPORT_TOLERANCE_INVALID")
    cell_teacher.validate()
    triangle_map = {int(triangle.triangle_id): triangle for triangle in triangles}
    if len(triangle_map) != len(triangles):
        raise ValueError("FIT8_PATCH_FACE_SUPPORT_DUPLICATE_TRIANGLE_ID")
    center = np.asarray(cell_teacher.cell.center_xyz, dtype=np.float64)
    half_extent = float(cell_teacher.cell.half_extent)
    rows: list[PatchFaceSupportRowV1] = []
    for patch in cell_teacher.local_patch_lower_bound:
        missing = [tid for tid in patch.source_triangle_ids if tid not in triangle_map]
        if missing:
            raise ValueError("FIT8_PATCH_FACE_SUPPORT_TRIANGLE_MISSING")
        faces = tuple(
            face_index
            for face_index in range(6)
            if any(
                triangle_touches_cell_face_v1(
                    triangle_map[tid].vertices(), center, half_extent, face_index, geometry_tolerance
                )
                for tid in patch.source_triangle_ids
            )
        )
        rows.append(PatchFaceSupportRowV1(patch.patch_local_id, faces))
    report = PatchFaceSupportReportV1(SCHEMA_VERSION, tuple(rows))
    report.validate()
    return report
