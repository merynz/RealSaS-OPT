from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

SCHEMA_VERSION = "RealSaS.PolygonTriangulationQualification.v1"


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    if a == b:
        raise ValueError("TRIANGULATION_DEGENERATE_EDGE")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class PolygonTriangulationQualificationReportV1:
    schema_version: str
    polygon_count: int
    candidate_triangle_count: int
    triangle_count_mismatch_polygons: tuple[int, ...]
    triangle_vertex_scope_violation_polygons: tuple[int, ...]
    boundary_incidence_violation_polygons: tuple[int, ...]
    internal_diagonal_incidence_violation_polygons: tuple[int, ...]
    internal_diagonal_alias_authored_boundary_keys: tuple[tuple[int, int], ...]
    internal_diagonal_reused_across_polygons_keys: tuple[tuple[int, int], ...]

    @property
    def topology_qualified(self) -> bool:
        return not any(
            (
                self.triangle_count_mismatch_polygons,
                self.triangle_vertex_scope_violation_polygons,
                self.boundary_incidence_violation_polygons,
                self.internal_diagonal_incidence_violation_polygons,
                self.internal_diagonal_alias_authored_boundary_keys,
                self.internal_diagonal_reused_across_polygons_keys,
            )
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("TRIANGULATION_SCHEMA_INVALID")
        if self.polygon_count < 0 or self.candidate_triangle_count < 0:
            raise ValueError("TRIANGULATION_NEGATIVE_COUNT")


def qualify_polygon_triangulation_topology_v1(
    source_polygons: Sequence[Sequence[int]],
    triangles_by_polygon: Mapping[int, Sequence[Sequence[int]]],
) -> PolygonTriangulationQualificationReportV1:
    """Check topology preservation of an already-proposed polygon triangulation.

    The court is intentionally combinatorial. Geometry/coverage/normal/depth and
    authored renderer parity are separate qualification layers.

    A derived internal diagonal is polygon-local evidence. Reusing its vertex-pair
    identity in another polygon, or aliasing an authored surface boundary edge,
    would collapse independent edge incidence in an ordinary indexed triangle mesh
    and can manufacture non-manifold topology.
    """
    if not source_polygons:
        raise ValueError("TRIANGULATION_SOURCE_EMPTY")

    source_boundary_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    local_boundaries: list[set[tuple[int, int]]] = []
    canonical_polygons: list[tuple[int, ...]] = []
    for polygon_id, raw_polygon in enumerate(source_polygons):
        polygon = tuple(map(int, raw_polygon))
        if len(polygon) < 3 or len(set(polygon)) != len(polygon) or min(polygon) < 0:
            raise ValueError("TRIANGULATION_SOURCE_POLYGON_INVALID")
        canonical_polygons.append(polygon)
        boundary = {
            _edge_key(a, b)
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        }
        local_boundaries.append(boundary)
        for edge in boundary:
            source_boundary_owners[edge].add(polygon_id)

    count_mismatch: list[int] = []
    vertex_scope_violation: list[int] = []
    boundary_incidence_violation: list[int] = []
    diagonal_incidence_violation: list[int] = []
    diagonal_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    candidate_triangle_count = 0

    for polygon_id, polygon in enumerate(canonical_polygons):
        allowed_vertices = set(polygon)
        triangles = tuple(
            tuple(map(int, triangle))
            for triangle in triangles_by_polygon.get(polygon_id, ())
        )
        candidate_triangle_count += len(triangles)
        if len(triangles) != len(polygon) - 2:
            count_mismatch.append(polygon_id)

        edge_counts: Counter[tuple[int, int]] = Counter()
        scope_bad = False
        for triangle in triangles:
            if (
                len(triangle) != 3
                or len(set(triangle)) != 3
                or any(vertex not in allowed_vertices for vertex in triangle)
            ):
                scope_bad = True
                continue
            for a, b in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge_counts[_edge_key(a, b)] += 1
        if scope_bad:
            vertex_scope_violation.append(polygon_id)

        boundary = local_boundaries[polygon_id]
        if any(edge_counts[edge] != 1 for edge in boundary):
            boundary_incidence_violation.append(polygon_id)

        diagonals = {
            edge: count
            for edge, count in edge_counts.items()
            if edge not in boundary
        }
        if any(count != 2 for count in diagonals.values()):
            diagonal_incidence_violation.append(polygon_id)
        for edge in diagonals:
            diagonal_owners[edge].add(polygon_id)

    alias_boundary = tuple(
        sorted(edge for edge in diagonal_owners if edge in source_boundary_owners)
    )
    reused_across_polygons = tuple(
        sorted(edge for edge, owners in diagonal_owners.items() if len(owners) > 1)
    )

    report = PolygonTriangulationQualificationReportV1(
        schema_version=SCHEMA_VERSION,
        polygon_count=len(canonical_polygons),
        candidate_triangle_count=candidate_triangle_count,
        triangle_count_mismatch_polygons=tuple(count_mismatch),
        triangle_vertex_scope_violation_polygons=tuple(vertex_scope_violation),
        boundary_incidence_violation_polygons=tuple(boundary_incidence_violation),
        internal_diagonal_incidence_violation_polygons=tuple(diagonal_incidence_violation),
        internal_diagonal_alias_authored_boundary_keys=alias_boundary,
        internal_diagonal_reused_across_polygons_keys=reused_across_polygons,
    )
    report.validate()
    return report
