from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import (
    ExtractionCellTeacherV1,
    SourceTriangleV1,
)

SCHEMA_VERSION = "RealSaS.CellSupportedPatchQualification.v1"


@dataclass(frozen=True)
class CellSupportedAdjacencyPolicyV1:
    geometry_tolerance: float
    provenance: str
    witness_set_complete: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.geometry_tolerance) or self.geometry_tolerance <= 0.0:
            raise ValueError("CELL_SUPPORTED_PATCH_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("CELL_SUPPORTED_PATCH_PROVENANCE_MISSING")
        if not self.witness_set_complete:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_SET_INCOMPLETE")


@dataclass(frozen=True)
class TriangleSharedEdgeWitnessV1:
    triangle_a: int
    triangle_b: int
    endpoint0_xyz: tuple[float, float, float]
    endpoint1_xyz: tuple[float, float, float]
    provenance: str

    @property
    def pair(self) -> tuple[int, int]:
        return (self.triangle_a, self.triangle_b)

    def validate(self, tolerance: float) -> None:
        if self.triangle_a < 0 or self.triangle_b < 0 or self.triangle_a >= self.triangle_b:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_PAIR_INVALID")
        p0 = np.asarray(self.endpoint0_xyz, dtype=np.float64)
        p1 = np.asarray(self.endpoint1_xyz, dtype=np.float64)
        if p0.shape != (3,) or p1.shape != (3,) or not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_EDGE_INVALID")
        if float(np.linalg.norm(p1 - p0)) <= tolerance:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_EDGE_DEGENERATE")
        if not self.provenance:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_PROVENANCE_MISSING")


@dataclass(frozen=True)
class CellSupportedPatchV1:
    patch_local_id: int
    parent_lower_bound_patch_id: int
    source_triangle_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.patch_local_id < 0 or self.parent_lower_bound_patch_id < 0:
            raise ValueError("CELL_SUPPORTED_PATCH_ID_INVALID")
        if not self.source_triangle_ids:
            raise ValueError("CELL_SUPPORTED_PATCH_EMPTY")
        if tuple(sorted(set(self.source_triangle_ids))) != self.source_triangle_ids:
            raise ValueError("CELL_SUPPORTED_PATCH_TRIANGLE_IDS_NOT_CANONICAL")


@dataclass(frozen=True)
class CellSupportedPatchQualificationV1:
    schema_version: str
    policy: CellSupportedAdjacencyPolicyV1
    patches: tuple[CellSupportedPatchV1, ...]
    positive_support_pairs: tuple[tuple[int, int], ...]
    point_touch_pairs: tuple[tuple[int, int], ...]
    outside_cell_pairs: tuple[tuple[int, int], ...]

    def validate(self, teacher: ExtractionCellTeacherV1) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("CELL_SUPPORTED_PATCH_SCHEMA_INVALID")
        self.policy.validate()
        if [patch.patch_local_id for patch in self.patches] != list(range(len(self.patches))):
            raise ValueError("CELL_SUPPORTED_PATCH_IDS_NOT_CONTIGUOUS")
        for patch in self.patches:
            patch.validate()
        covered = tuple(sorted(triangle_id for patch in self.patches for triangle_id in patch.source_triangle_ids))
        if covered != teacher.intersecting_triangle_ids:
            raise ValueError("CELL_SUPPORTED_PATCH_PARTITION_COVERAGE_MISMATCH")
        if len(covered) != len(set(covered)):
            raise ValueError("CELL_SUPPORTED_PATCH_PARTITION_OVERLAP")
        for rows in (self.positive_support_pairs, self.point_touch_pairs, self.outside_cell_pairs):
            if tuple(sorted(set(rows))) != rows:
                raise ValueError("CELL_SUPPORTED_PATCH_PAIR_SET_NOT_CANONICAL")


def _point_matches(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    return float(np.linalg.norm(a - b)) <= tolerance


def _triangle_has_edge(triangle: SourceTriangleV1, p0: np.ndarray, p1: np.ndarray, tolerance: float) -> bool:
    vertices = triangle.vertices()
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = vertices[i], vertices[j]
        if (_point_matches(a, p0, tolerance) and _point_matches(b, p1, tolerance)) or (
            _point_matches(a, p1, tolerance) and _point_matches(b, p0, tolerance)
        ):
            return True
    return False


def _segment_aabb_support_kind(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    half_extent: float,
    tolerance: float,
) -> str:
    """Return POSITIVE, POINT, or OUTSIDE for segment support inside one cell."""
    lo = center - float(half_extent)
    hi = center + float(half_extent)
    direction = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= tolerance:
            if p0[axis] < lo[axis] - tolerance or p0[axis] > hi[axis] + tolerance:
                return "OUTSIDE"
            continue
        a = float((lo[axis] - p0[axis]) / direction[axis])
        b = float((hi[axis] - p0[axis]) / direction[axis])
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
        if t0 > t1 + tolerance:
            return "OUTSIDE"
    q0 = p0 + t0 * direction
    q1 = p0 + t1 * direction
    if float(np.linalg.norm(q1 - q0)) > tolerance:
        return "POSITIVE"
    return "POINT"


def qualify_cell_supported_patches_v1(
    teacher: ExtractionCellTeacherV1,
    triangles: Sequence[SourceTriangleV1],
    adjacency_witnesses: Sequence[TriangleSharedEdgeWitnessV1],
    *,
    policy: CellSupportedAdjacencyPolicyV1,
) -> CellSupportedPatchQualificationV1:
    """Split coarse cell-local patches using only shared edges supported inside the cell.

    `teacher.local_patch_lower_bound` uses source adjacency restricted to triangles
    intersecting the cell. That can over-connect triangles whose common authored
    edge lies completely outside the cell. This qualifier never merges lower-bound
    patches; it only splits them when the qualified shared-edge witness has no
    positive-length support inside the extraction cell.

    No component/material/owner equality is used as adjacency authority.
    """
    teacher.validate()
    policy.validate()
    tolerance = float(policy.geometry_tolerance)
    triangle_map = {int(triangle.triangle_id): triangle for triangle in triangles}
    if len(triangle_map) != len(triangles):
        raise ValueError("CELL_SUPPORTED_PATCH_DUPLICATE_TRIANGLE_ID")
    for triangle_id in teacher.intersecting_triangle_ids:
        if triangle_id not in triangle_map:
            raise ValueError("CELL_SUPPORTED_PATCH_TRIANGLE_RECORD_MISSING")

    parent_by_triangle: dict[int, int] = {}
    for patch in teacher.local_patch_lower_bound:
        for triangle_id in patch.source_triangle_ids:
            if triangle_id in parent_by_triangle:
                raise ValueError("CELL_SUPPORTED_PATCH_LOWER_BOUND_OVERLAP")
            parent_by_triangle[int(triangle_id)] = int(patch.patch_local_id)

    seen_pairs: set[tuple[int, int]] = set()
    positive_pairs: list[tuple[int, int]] = []
    point_pairs: list[tuple[int, int]] = []
    outside_pairs: list[tuple[int, int]] = []
    allowed = set(map(int, teacher.intersecting_triangle_ids))
    center = np.asarray(teacher.cell.center_xyz, dtype=np.float64)

    for witness in adjacency_witnesses:
        witness.validate(tolerance)
        pair = witness.pair
        if pair in seen_pairs:
            raise ValueError("CELL_SUPPORTED_PATCH_DUPLICATE_WITNESS_PAIR")
        seen_pairs.add(pair)
        a, b = pair
        if a not in triangle_map or b not in triangle_map:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_TRIANGLE_UNKNOWN")
        p0 = np.asarray(witness.endpoint0_xyz, dtype=np.float64)
        p1 = np.asarray(witness.endpoint1_xyz, dtype=np.float64)
        if not _triangle_has_edge(triangle_map[a], p0, p1, tolerance) or not _triangle_has_edge(
            triangle_map[b], p0, p1, tolerance
        ):
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_NOT_SHARED_TRIANGLE_EDGE")
        if a not in allowed or b not in allowed:
            continue
        if parent_by_triangle[a] != parent_by_triangle[b]:
            raise ValueError("CELL_SUPPORTED_PATCH_WITNESS_MERGES_LOWER_BOUND_PATCHES")
        kind = _segment_aabb_support_kind(p0, p1, center, teacher.cell.half_extent, tolerance)
        if kind == "POSITIVE":
            positive_pairs.append(pair)
        elif kind == "POINT":
            point_pairs.append(pair)
        else:
            outside_pairs.append(pair)

    adjacency: dict[int, set[int]] = {triangle_id: set() for triangle_id in allowed}
    for a, b in positive_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    unseen = set(allowed)
    groups: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        groups.append(tuple(sorted(group)))
    groups.sort(key=lambda group: (group[0], len(group)))

    patches: list[CellSupportedPatchV1] = []
    for patch_local_id, group in enumerate(groups):
        parents = {parent_by_triangle[triangle_id] for triangle_id in group}
        if len(parents) != 1:
            raise ValueError("CELL_SUPPORTED_PATCH_RESULT_MERGES_LOWER_BOUND_PATCHES")
        patches.append(
            CellSupportedPatchV1(
                patch_local_id=patch_local_id,
                parent_lower_bound_patch_id=next(iter(parents)),
                source_triangle_ids=group,
            )
        )

    result = CellSupportedPatchQualificationV1(
        schema_version=SCHEMA_VERSION,
        policy=policy,
        patches=tuple(patches),
        positive_support_pairs=tuple(sorted(positive_pairs)),
        point_touch_pairs=tuple(sorted(point_pairs)),
        outside_cell_pairs=tuple(sorted(outside_pairs)),
    )
    result.validate(teacher)
    return result
