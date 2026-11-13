"""Geometric/topological identity for artist meshes with render-vertex splits.

Render/import vertex identity is intentionally NOT topology identity. glTF and
DCC transports may duplicate a geometric vertex at UV, material, tangent or
normal seams. This module builds a fail-closed quotient using exact source-space
position within an authored mesh owner, while preserving the raw render vertices
and face addresses unchanged.

No tolerance weld is performed. A quotient vertex may only combine raw vertices
that are referenced by faces, belong to the same face owner, and have exactly
equal finite float64 positions after signed-zero normalization.

The quotient is separately audited for edge manifoldness, orientation consistency
and vertex-link manifoldness, so coincident-but-distinct shells that touch at a
point cannot silently become a valid closed surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


def _mesh(vertices, faces, face_owner=None):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if (
        v.ndim != 2 or v.shape[1:] != (3,) or not len(v)
        or not np.isfinite(v).all()
        or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
        or np.min(f) < 0 or np.max(f) >= len(v)
    ):
        raise ValueError("GEOMETRIC_TOPOLOGY_FINITE_TRIANGULATED_MESH_REQUIRED")
    if face_owner is None:
        owner = np.zeros(len(f), dtype=np.int32)
    else:
        owner = np.asarray(face_owner, dtype=np.int32)
        if owner.shape != (len(f),) or np.any(owner < 0):
            raise ValueError("GEOMETRIC_TOPOLOGY_FACE_OWNER_LAYOUT")
    return v, f, owner


def _normalized_position_key(position: np.ndarray) -> bytes:
    p = np.asarray(position, dtype=np.float64).copy()
    p[p == 0.0] = 0.0
    return p.tobytes(order="C")


def _raw_vertex_owner(face_count: int, vertex_count: int, faces: np.ndarray, face_owner: np.ndarray) -> np.ndarray:
    owner = np.full(vertex_count, -1, dtype=np.int32)
    for fi in range(face_count):
        oid = int(face_owner[fi])
        for raw in faces[fi]:
            raw = int(raw)
            prior = int(owner[raw])
            if prior == -1:
                owner[raw] = oid
            elif prior != oid:
                raise ValueError("GEOMETRIC_TOPOLOGY_RAW_VERTEX_CROSSES_OWNERS")
    return owner


def _edge_topology(faces: np.ndarray):
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    edges, inverse, incidence = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.ones(len(edges), dtype=np.bool_)
    for edge_id in np.flatnonzero(incidence == 2):
        rows = np.flatnonzero(inverse == edge_id)
        a, b = directed[rows[0]], directed[rows[1]]
        orientation[edge_id] = bool(a[0] == b[1] and a[1] == b[0])
    return (
        edges.astype(np.int32), incidence.astype(np.int32),
        incidence == 1, incidence > 2, orientation,
    )


def _face_components(vertex_count: int, faces: np.ndarray):
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        a, b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)

    referenced = np.unique(faces)
    for tri in faces:
        union(int(tri[0]), int(tri[1]))
        union(int(tri[1]), int(tri[2]))
        union(int(tri[2]), int(tri[0]))
    roots = [find(int(i)) for i in referenced]
    lut = {root: index for index, root in enumerate(sorted(set(roots)))}
    vertex_component = np.full(vertex_count, -1, dtype=np.int32)
    for vertex, root in zip(referenced, roots):
        vertex_component[int(vertex)] = lut[root]
    face_component = vertex_component[faces[:, 0]]
    if np.any(vertex_component[faces] != face_component[:, None]):
        raise ValueError("GEOMETRIC_TOPOLOGY_COMPONENT_INCONSISTENCY")
    return vertex_component, face_component.astype(np.int32)


def _vertex_link_manifold(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    """Closed-manifold vertex court: every referenced vertex link is one cycle."""
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for fi, tri in enumerate(faces):
        for vertex in tri:
            incident[int(vertex)].append(fi)

    valid = np.ones(vertex_count, dtype=np.bool_)
    for vertex, face_ids in enumerate(incident):
        if not face_ids:
            continue
        link_edges = []
        for fi in face_ids:
            tri = faces[fi]
            others = [int(x) for x in tri if int(x) != vertex]
            if len(others) != 2:
                valid[vertex] = False
                break
            link_edges.append(tuple(sorted(others)))
        if not valid[vertex]:
            continue
        degree: dict[int, int] = {}
        graph: dict[int, set[int]] = {}
        for a, b in link_edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        if not degree or any(value != 2 for value in degree.values()):
            valid[vertex] = False
            continue
        start = next(iter(degree))
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        valid[vertex] = len(seen) == len(degree)
    return valid


@dataclass(frozen=True)
class GeometricTopology:
    raw_to_topology_vertex: np.ndarray
    topology_vertices: np.ndarray
    topology_faces: np.ndarray
    topology_vertex_owner: np.ndarray
    unique_edge_index: np.ndarray
    edge_face_incidence_count: np.ndarray
    boundary_edge: np.ndarray
    nonmanifold_edge: np.ndarray
    edge_orientation_consistent: np.ndarray
    topology_vertex_component_id: np.ndarray
    face_component_id: np.ndarray
    vertex_link_manifold: np.ndarray
    exact_duplicate_collapse_count: int
    identity_sha256: str

    @property
    def closed_edge_manifold(self) -> bool:
        return bool(
            not np.any(self.boundary_edge)
            and not np.any(self.nonmanifold_edge)
            and np.all(self.edge_orientation_consistent[self.edge_face_incidence_count == 2])
        )

    @property
    def closed_vertex_manifold(self) -> bool:
        referenced = np.unique(self.topology_faces)
        return bool(np.all(self.vertex_link_manifold[referenced]))

    @property
    def sign_topology_valid(self) -> bool:
        return bool(self.closed_edge_manifold and self.closed_vertex_manifold)


def build_geometric_topology_v1(vertices, faces, face_owner=None) -> GeometricTopology:
    v, f, owner = _mesh(vertices, faces, face_owner)
    raw_owner = _raw_vertex_owner(len(f), len(v), f, owner)

    raw_to_topology = np.full(len(v), -1, dtype=np.int32)
    topology_vertices: list[np.ndarray] = []
    topology_owner: list[int] = []
    lookup: dict[tuple[int, bytes], int] = {}

    for raw in np.unique(f):
        raw = int(raw)
        oid = int(raw_owner[raw])
        if oid < 0:
            raise ValueError("GEOMETRIC_TOPOLOGY_REFERENCED_VERTEX_OWNER_MISSING")
        key = (oid, _normalized_position_key(v[raw]))
        top = lookup.get(key)
        if top is None:
            top = len(topology_vertices)
            lookup[key] = top
            topology_vertices.append(v[raw].copy())
            topology_owner.append(oid)
        raw_to_topology[raw] = top

    qv = np.asarray(topology_vertices, dtype=np.float64)
    qowner = np.asarray(topology_owner, dtype=np.int32)
    qf = raw_to_topology[f]
    if np.any(qf < 0):
        raise ValueError("GEOMETRIC_TOPOLOGY_FACE_REMAP_FAILED")
    if np.any(
        (qf[:, 0] == qf[:, 1])
        | (qf[:, 1] == qf[:, 2])
        | (qf[:, 2] == qf[:, 0])
    ):
        raise ValueError("GEOMETRIC_TOPOLOGY_TRIANGLE_COLLAPSE_AFTER_EXACT_QUOTIENT")

    edges, incidence, boundary, nonmanifold, orientation = _edge_topology(qf)
    vc, fc = _face_components(len(qv), qf)
    link = _vertex_link_manifold(len(qv), qf)

    identity = hashlib.sha256(
        qv.tobytes(order="C")
        + qf.astype(np.int32, copy=False).tobytes(order="C")
        + qowner.tobytes(order="C")
    ).hexdigest()

    referenced_raw = np.unique(f)
    return GeometricTopology(
        raw_to_topology_vertex=raw_to_topology,
        topology_vertices=qv,
        topology_faces=qf.astype(np.int32),
        topology_vertex_owner=qowner,
        unique_edge_index=edges,
        edge_face_incidence_count=incidence,
        boundary_edge=boundary.astype(np.bool_),
        nonmanifold_edge=nonmanifold.astype(np.bool_),
        edge_orientation_consistent=orientation.astype(np.bool_),
        topology_vertex_component_id=vc,
        face_component_id=fc,
        vertex_link_manifold=link.astype(np.bool_),
        exact_duplicate_collapse_count=int(len(referenced_raw) - len(qv)),
        identity_sha256=identity,
    )


def face_adjacency_from_geometric_topology_v1(vertices, faces, face_owner=None) -> np.ndarray:
    topology = build_geometric_topology_v1(vertices, faces, face_owner)
    edge_owner: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(topology.topology_faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_owner.setdefault(edge, []).append(face_index)
    pairs = set()
    for owners in edge_owner.values():
        if len(owners) == 2:
            a, b = sorted(owners)
            pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
