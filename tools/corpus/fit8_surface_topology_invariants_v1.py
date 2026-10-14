from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

SCHEMA_VERSION = "RealSaS.SurfaceTopologyInvariants.v1"


@dataclass(frozen=True)
class SurfaceComponentTopologyV1:
    component_id: int
    triangle_ids: tuple[int, ...]
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    boundary_edge_count: int
    boundary_graph_component_count: int
    boundary_loop_count: int | None
    nonmanifold_edge_count: int
    orientability_valid: bool
    orientable: bool | None
    genus_valid: bool
    genus: int | None

    def validate(self) -> None:
        if self.component_id < 0 or not self.triangle_ids:
            raise ValueError("SURFACE_TOPOLOGY_COMPONENT_INVALID")
        for name in (
            "vertex_count",
            "edge_count",
            "face_count",
            "boundary_edge_count",
            "boundary_graph_component_count",
            "nonmanifold_edge_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"SURFACE_TOPOLOGY_NEGATIVE:{name}")
        if self.boundary_loop_count is not None and self.boundary_loop_count < 0:
            raise ValueError("SURFACE_TOPOLOGY_BOUNDARY_LOOP_INVALID")
        if self.genus_valid != (self.genus is not None):
            raise ValueError("SURFACE_TOPOLOGY_GENUS_VALIDITY_MISMATCH")
        if self.genus is not None and self.genus < 0:
            raise ValueError("SURFACE_TOPOLOGY_GENUS_NEGATIVE")


@dataclass(frozen=True)
class SurfaceTopologyReportV1:
    schema_version: str
    component_count: int
    components: tuple[SurfaceComponentTopologyV1, ...]
    total_boundary_edges: int
    total_nonmanifold_edges: int

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("SURFACE_TOPOLOGY_SCHEMA_INVALID")
        if self.component_count != len(self.components):
            raise ValueError("SURFACE_TOPOLOGY_COMPONENT_COUNT_MISMATCH")
        if [component.component_id for component in self.components] != list(range(len(self.components))):
            raise ValueError("SURFACE_TOPOLOGY_COMPONENT_IDS_INVALID")
        for component in self.components:
            component.validate()
        if self.total_boundary_edges != sum(component.boundary_edge_count for component in self.components):
            raise ValueError("SURFACE_TOPOLOGY_BOUNDARY_TOTAL_MISMATCH")
        if self.total_nonmanifold_edges != sum(component.nonmanifold_edge_count for component in self.components):
            raise ValueError("SURFACE_TOPOLOGY_NONMANIFOLD_TOTAL_MISMATCH")


def _edge_key(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError("SURFACE_TOPOLOGY_DEGENERATE_EDGE")
    return (a, b) if a < b else (b, a)


def _boundary_graph(boundary_edges: Sequence[tuple[int, int]]) -> tuple[int, int | None]:
    if not boundary_edges:
        return 0, 0
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        graph[a].add(b)
        graph[b].add(a)
    unseen = set(graph)
    component_count = 0
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component_count += 1
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    loop_count = component_count if all(len(neighbors) == 2 for neighbors in graph.values()) else None
    return component_count, loop_count


def _orientability(
    face_ids: Sequence[int],
    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]],
    nonmanifold_edge_count: int,
) -> tuple[bool, bool | None]:
    if nonmanifold_edge_count:
        return False, None
    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rows in edge_incidence.values():
        if len(rows) != 2:
            continue
        (face_a, sign_a), (face_b, sign_b) = rows
        relation = -sign_a * sign_b
        neighbors[face_a].append((face_b, relation))
        neighbors[face_b].append((face_a, relation))
    assignment: dict[int, int] = {}
    for root in face_ids:
        if root in assignment:
            continue
        assignment[root] = 1
        queue = deque([root])
        while queue:
            face = queue.popleft()
            for neighbor, relation in neighbors.get(face, ()): 
                expected = assignment[face] * relation
                if neighbor in assignment:
                    if assignment[neighbor] != expected:
                        return True, False
                else:
                    assignment[neighbor] = expected
                    queue.append(neighbor)
    return True, True


def compute_surface_topology_invariants_v1(
    faces: Iterable[Sequence[int]],
) -> SurfaceTopologyReportV1:
    """Compute topology invariants on an already-qualified triangle carrier.

    This function does not weld render seams, triangulate source polygons or repair
    topology. Those transformations require separate authority. Genus is emitted
    only when orientability and boundary-loop preconditions are valid.
    """
    face_rows = tuple(tuple(map(int, face)) for face in faces)
    if not face_rows:
        raise ValueError("SURFACE_TOPOLOGY_EMPTY")
    for face in face_rows:
        if len(face) != 3 or len(set(face)) != 3 or min(face) < 0:
            raise ValueError("SURFACE_TOPOLOGY_FACE_INVALID")

    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    triangle_neighbors: dict[int, set[int]] = defaultdict(set)
    for face_id, (a, b, c) in enumerate(face_rows):
        for u, v in ((a, b), (b, c), (c, a)):
            key = _edge_key(u, v)
            sign = 1 if (u, v) == key else -1
            for prior_face, _ in edge_incidence[key]:
                triangle_neighbors[face_id].add(prior_face)
                triangle_neighbors[prior_face].add(face_id)
            edge_incidence[key].append((face_id, sign))

    unseen = set(range(len(face_rows)))
    groups: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        group: list[int] = []
        while stack:
            face_id = stack.pop()
            group.append(face_id)
            for neighbor in triangle_neighbors.get(face_id, ()): 
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        groups.append(tuple(sorted(group)))
    groups.sort(key=lambda group: group[0])

    components: list[SurfaceComponentTopologyV1] = []
    for component_id, group in enumerate(groups):
        group_set = set(group)
        vertices = {vertex for face_id in group for vertex in face_rows[face_id]}
        local_edges = {
            edge: [row for row in rows if row[0] in group_set]
            for edge, rows in edge_incidence.items()
            if any(face_id in group_set for face_id, _ in rows)
        }
        boundary_edges = [edge for edge, rows in local_edges.items() if len(rows) == 1]
        nonmanifold_edge_count = sum(len(rows) > 2 for rows in local_edges.values())
        boundary_graph_components, boundary_loops = _boundary_graph(boundary_edges)
        orientability_valid, orientable = _orientability(
            group,
            local_edges,
            nonmanifold_edge_count,
        )
        vertex_count = len(vertices)
        edge_count = len(local_edges)
        face_count = len(group)
        euler = vertex_count - edge_count + face_count
        genus = None
        genus_valid = False
        if orientability_valid and orientable and boundary_loops is not None:
            numerator = 2 - boundary_loops - euler
            if numerator >= 0 and numerator % 2 == 0:
                genus = numerator // 2
                genus_valid = True
        components.append(
            SurfaceComponentTopologyV1(
                component_id=component_id,
                triangle_ids=group,
                vertex_count=vertex_count,
                edge_count=edge_count,
                face_count=face_count,
                euler_characteristic=euler,
                boundary_edge_count=len(boundary_edges),
                boundary_graph_component_count=boundary_graph_components,
                boundary_loop_count=boundary_loops,
                nonmanifold_edge_count=nonmanifold_edge_count,
                orientability_valid=orientability_valid,
                orientable=orientable,
                genus_valid=genus_valid,
                genus=genus,
            )
        )

    report = SurfaceTopologyReportV1(
        schema_version=SCHEMA_VERSION,
        component_count=len(components),
        components=tuple(components),
        total_boundary_edges=sum(component.boundary_edge_count for component in components),
        total_nonmanifold_edges=sum(component.nonmanifold_edge_count for component in components),
    )
    report.validate()
    return report
