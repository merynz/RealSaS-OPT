from __future__ import annotations

"""Component-scoped geometry/sign authority for artist-authored meshes.

This producer-side module intentionally does not import the Corpus400 structural
measurement helper. It builds its own owner-preserving exact-position quotient,
separates surface connectivity from sign/manifold connectivity, and emits explicit
face/component validity. Cross-owner coincidence is never an implicit weld.

Scientific boundary:
- unsigned source-surface distance may remain authoritative on open surfaces;
- sign/occupancy is authoritative only for independently closed manifold components;
- degenerate/collapsed faces are masked, never interpreted as negative space;
- component IDs are local addresses, not semantic/mechanical identities.
"""

from dataclasses import dataclass
import hashlib
import numpy as np


def _position_key(value: np.ndarray) -> bytes:
    value = np.asarray(value, dtype=np.float64).copy()
    value[value == 0.0] = 0.0
    return value.tobytes(order="C")


def _validate_mesh(vertices, faces, face_owner):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    owner = np.asarray(face_owner, dtype=np.int32)
    if (
        v.ndim != 2
        or v.shape[1:] != (3,)
        or not len(v)
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1:] != (3,)
        or not len(f)
        or np.any(f < 0)
        or np.any(f >= len(v))
        or owner.shape != (len(f),)
        or np.any(owner < 0)
    ):
        raise ValueError("COMPONENT_GEOMETRY_FINITE_TRIANGULATED_MESH_REQUIRED")
    return v, f, owner


def _raw_vertex_owner(vertex_count, faces, owner):
    result = np.full(vertex_count, -1, dtype=np.int32)
    for fi, tri in enumerate(faces):
        oid = int(owner[fi])
        for raw in tri:
            raw = int(raw)
            previous = int(result[raw])
            if previous == -1:
                result[raw] = oid
            elif previous != oid:
                raise ValueError("COMPONENT_GEOMETRY_RAW_VERTEX_CROSSES_OWNERS")
    return result


def _quotient(vertices, faces, owner):
    raw_owner = _raw_vertex_owner(len(vertices), faces, owner)
    raw_to_q = np.full(len(vertices), -1, dtype=np.int32)
    qv = []
    qowner = []
    members = []
    lookup = {}
    for raw in np.unique(faces):
        raw = int(raw)
        oid = int(raw_owner[raw])
        key = (oid, _position_key(vertices[raw]))
        qid = lookup.get(key)
        if qid is None:
            qid = len(qv)
            lookup[key] = qid
            qv.append(vertices[raw].copy())
            qowner.append(oid)
            members.append([])
        raw_to_q[raw] = qid
        members[qid].append(raw)
    qv = np.asarray(qv, dtype=np.float64)
    qf = raw_to_q[faces]
    if np.any(qf < 0):
        raise ValueError("COMPONENT_GEOMETRY_QUOTIENT_REMAP")
    return (
        raw_to_q,
        qv,
        qf.astype(np.int32),
        np.asarray(qowner, dtype=np.int32),
        tuple(tuple(row) for row in members),
    )


def _face_double_area(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )


def _edge_rows(faces):
    table = {}
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            table.setdefault(key, []).append((fi, (a, b)))
    return table


def _components(face_count, graph, enabled=None):
    result = np.full(face_count, -1, dtype=np.int32)
    enabled_mask = np.ones(face_count, dtype=np.bool_) if enabled is None else np.asarray(enabled, dtype=np.bool_)
    cid = 0
    for start in range(face_count):
        if not enabled_mask[start] or result[start] >= 0:
            continue
        result[start] = cid
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if enabled_mask[neighbor] and result[neighbor] < 0:
                    result[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    return result


def _surface_components(faces):
    graph = [set() for _ in range(len(faces))]
    for rows in _edge_rows(faces).values():
        ids = sorted({int(fi) for fi, _ in rows})
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return _components(len(faces), graph)


def _sign_candidate_components(faces, face_valid):
    graph = [set() for _ in range(len(faces))]
    for rows in _edge_rows(faces).values():
        if len(rows) != 2:
            continue
        (left, d0), (right, d1) = rows
        if not (face_valid[left] and face_valid[right]):
            continue
        if d0[0] == d1[1] and d0[1] == d1[0]:
            graph[left].add(right)
            graph[right].add(left)
    return _components(len(faces), graph, enabled=face_valid)


def _vertex_link_closed(vertices_count, faces):
    if not len(faces):
        return False
    incident = {}
    for fi, tri in enumerate(faces):
        for vertex in tri:
            incident.setdefault(int(vertex), []).append(fi)
    for vertex, face_ids in incident.items():
        graph = {}
        degree = {}
        for fi in face_ids:
            tri = faces[fi]
            other = [int(x) for x in tri if int(x) != vertex]
            if len(other) != 2:
                return False
            a, b = other
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        if not degree or any(value != 2 for value in degree.values()):
            return False
        start = next(iter(degree))
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(seen) != len(degree):
            return False
    return True


def _component_validity(faces, sign_component, face_valid):
    global_edges = _edge_rows(faces)
    rows = []
    if not np.any(sign_component >= 0):
        return rows
    for cid in range(int(sign_component.max()) + 1):
        face_ids = np.flatnonzero(sign_component == cid)
        face_set = set(map(int, face_ids))
        boundary = 0
        nonmanifold = 0
        orientation_conflict = 0
        foreign_incidence = 0
        for edge, global_rows in global_edges.items():
            local = [(fi, direction) for fi, direction in global_rows if fi in face_set]
            if not local:
                continue
            if len(local) != 2:
                boundary += 1
            if len(global_rows) != 2:
                if len(global_rows) > 2:
                    nonmanifold += 1
                foreign_incidence += int(len(global_rows) != len(local))
            if len(local) == 2:
                d0, d1 = local[0][1], local[1][1]
                if not (d0[0] == d1[1] and d0[1] == d1[0]):
                    orientation_conflict += 1
        vertex_link = _vertex_link_closed(
            int(np.max(faces)) + 1,
            faces[face_ids],
        )
        all_faces_valid = bool(np.all(face_valid[face_ids]))
        closed = bool(
            len(face_ids)
            and all_faces_valid
            and boundary == 0
            and nonmanifold == 0
            and orientation_conflict == 0
            and foreign_incidence == 0
            and vertex_link
        )
        rows.append(
            {
                "component_id": cid,
                "face_count": int(len(face_ids)),
                "all_faces_nondegenerate": all_faces_valid,
                "boundary_edge_count": int(boundary),
                "nonmanifold_edge_count": int(nonmanifold),
                "orientation_conflict_edge_count": int(orientation_conflict),
                "foreign_edge_incidence_count": int(foreign_incidence),
                "vertex_link_closed": bool(vertex_link),
                "sign_authority_candidate": closed,
            }
        )
    return rows


@dataclass(frozen=True)
class ComponentGeometryAuthority:
    raw_to_topology_vertex: np.ndarray
    topology_vertices: np.ndarray
    topology_faces: np.ndarray
    topology_vertex_owner: np.ndarray
    raw_members: tuple[tuple[int, ...], ...]
    face_valid: np.ndarray
    surface_component_id: np.ndarray
    sign_component_id: np.ndarray
    sign_component_valid: np.ndarray
    sign_components: tuple[dict, ...]
    exact_duplicate_collapse_count: int
    identity_sha256: str

    def sign_valid_for_face(self, face_id):
        face_id = np.asarray(face_id, dtype=np.int64)
        component = self.sign_component_id[face_id]
        out = np.zeros(component.shape, dtype=np.bool_)
        valid = component >= 0
        out[valid] = self.sign_component_valid[component[valid]]
        return out


def build_component_geometry_authority_v1(vertices, faces, face_owner) -> ComponentGeometryAuthority:
    v, f, owner = _validate_mesh(vertices, faces, face_owner)
    raw_to_q, qv, qf, qowner, members = _quotient(v, f, owner)
    raw_area = _face_double_area(v, f)
    q_area = _face_double_area(qv, qf)
    collapsed = (
        (qf[:, 0] == qf[:, 1])
        | (qf[:, 1] == qf[:, 2])
        | (qf[:, 2] == qf[:, 0])
    )
    face_valid = (
        np.isfinite(raw_area)
        & (raw_area > 1.0e-12)
        & ~collapsed
        & np.isfinite(q_area)
        & (q_area > 1.0e-12)
    )
    surface_component = _surface_components(qf)
    sign_component = _sign_candidate_components(qf, face_valid)
    component_rows = _component_validity(qf, sign_component, face_valid)
    component_valid = np.asarray(
        [row["sign_authority_candidate"] for row in component_rows], dtype=np.bool_
    )
    referenced = np.unique(f)
    identity = hashlib.sha256(
        qv.tobytes(order="C")
        + qf.astype(np.int32, copy=False).tobytes(order="C")
        + qowner.tobytes(order="C")
        + surface_component.tobytes(order="C")
        + sign_component.tobytes(order="C")
        + component_valid.tobytes(order="C")
    ).hexdigest()
    return ComponentGeometryAuthority(
        raw_to_topology_vertex=raw_to_q,
        topology_vertices=qv,
        topology_faces=qf,
        topology_vertex_owner=qowner,
        raw_members=members,
        face_valid=face_valid.astype(np.bool_),
        surface_component_id=surface_component.astype(np.int32),
        sign_component_id=sign_component.astype(np.int32),
        sign_component_valid=component_valid,
        sign_components=tuple(component_rows),
        exact_duplicate_collapse_count=int(len(referenced) - len(qv)),
        identity_sha256=identity,
    )
