from __future__ import annotations

"""Qualified mechanical surface topology for modular authored meshes.

This authority is deliberately separate from render identity, appearance addressing,
and SDF/sign topology. It starts from an owner-preserving exact-position quotient and
may additionally identify vertices across authored owners only when a full boundary
edge is matched in reverse orientation and both endpoint trajectories are identical
through the sealed posed intervention tensor.

Point-only coincidence, same-orientation overlap, ambiguous >2-way edge incidence,
unknown trajectories, or trajectory drift never authorize a cross-owner weld.
"""

from dataclasses import dataclass
import hashlib
import numpy as np


def _key(value: np.ndarray) -> bytes:
    value = np.asarray(value, dtype=np.float64).copy()
    value[value == 0.0] = 0.0
    return value.tobytes(order="C")


def _validate(vertices, faces, face_owner, posed_vertices):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    owner = np.asarray(face_owner, dtype=np.int32)
    posed = np.asarray(posed_vertices, dtype=np.float64)
    if (
        v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
        or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
        or np.any(f < 0) or np.any(f >= len(v))
        or owner.shape != (len(f),) or np.any(owner < 0)
        or posed.ndim != 3 or posed.shape[1:] != v.shape or not len(posed)
        or not np.isfinite(posed).all()
    ):
        raise ValueError("MECHANICAL_SURFACE_TOPOLOGY_INPUT_LAYOUT")
    return v, f, owner, posed


def _raw_owner(vertex_count: int, faces: np.ndarray, owner: np.ndarray) -> np.ndarray:
    out = np.full(vertex_count, -1, dtype=np.int32)
    for fi, tri in enumerate(faces):
        oid = int(owner[fi])
        for raw in tri:
            raw = int(raw)
            if out[raw] == -1:
                out[raw] = oid
            elif int(out[raw]) != oid:
                raise ValueError("MECHANICAL_SURFACE_RAW_VERTEX_CROSSES_OWNERS")
    return out


def _owner_exact_quotient(v, f, owner):
    rv_owner = _raw_owner(len(v), f, owner)
    raw_to_q = np.full(len(v), -1, dtype=np.int32)
    qv, qowner, members, lookup = [], [], [], {}
    for raw in np.unique(f):
        raw = int(raw)
        oid = int(rv_owner[raw])
        token = (oid, _key(v[raw]))
        qid = lookup.get(token)
        if qid is None:
            qid = len(qv)
            lookup[token] = qid
            qv.append(v[raw].copy())
            qowner.append(oid)
            members.append([])
        raw_to_q[raw] = qid
        members[qid].append(raw)
    qv = np.asarray(qv, dtype=np.float64)
    qf = raw_to_q[f]
    if np.any(qf < 0):
        raise ValueError("MECHANICAL_SURFACE_OWNER_QUOTIENT_REMAP")
    return raw_to_q, qv, qf.astype(np.int32), np.asarray(qowner, np.int32), tuple(tuple(x) for x in members)


def _edge_table(faces):
    table = {}
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            table.setdefault((min(a, b), max(a, b)), []).append((fi, (a, b)))
    return table


def _trajectory_drift(posed, raw_members_a, raw_members_b, scale):
    members = sorted(set(map(int, raw_members_a)) | set(map(int, raw_members_b)))
    if len(members) < 2:
        return 0.0
    p = posed[:, members, :]
    ref = p[:, :1, :]
    drift = float(np.max(np.linalg.norm(p - ref, axis=-1), initial=0.0))
    return drift / max(scale, 1.0e-12)


def _face_components(faces):
    graph = [set() for _ in range(len(faces))]
    for rows in _edge_table(faces).values():
        ids = sorted({int(fi) for fi, _ in rows})
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                graph[a].add(b); graph[b].add(a)
    out = np.full(len(faces), -1, dtype=np.int32)
    cid = 0
    for start in range(len(faces)):
        if out[start] >= 0:
            continue
        out[start] = cid
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in graph[cur]:
                if out[nxt] < 0:
                    out[nxt] = cid; stack.append(nxt)
        cid += 1
    return out


def _double_area(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


@dataclass(frozen=True)
class MechanicalSurfaceTopologyAuthority:
    raw_to_mechanical_vertex: np.ndarray
    mechanical_vertices_source_world: np.ndarray
    mechanical_faces: np.ndarray
    face_owner: np.ndarray
    face_valid: np.ndarray
    surface_component_id: np.ndarray
    mechanical_vertex_owner_offsets: np.ndarray
    mechanical_vertex_owner_values: np.ndarray
    qualified_cross_owner_seam_count: int
    qualified_cross_owner_seam_edges: tuple[dict, ...]
    maximum_qualified_seam_trajectory_drift_over_scale: float
    identity_sha256: str


def build_mechanical_surface_topology_authority_v1(
    vertices,
    faces,
    face_owner,
    posed_vertices,
    *,
    trajectory_tolerance_over_scale: float = 1.0e-9,
) -> MechanicalSurfaceTopologyAuthority:
    v, f, owner, posed = _validate(vertices, faces, face_owner, posed_vertices)
    raw_to_q, qv, qf, qowner, members = _owner_exact_quotient(v, f, owner)
    scale = max(float(np.linalg.norm(v.max(0) - v.min(0))), 1.0e-12)

    parent = np.arange(len(qv), dtype=np.int32)
    def find(x):
        x = int(x)
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    boundary = []
    for edge, rows in _edge_table(qf).items():
        if len(rows) == 1:
            fi, directed = rows[0]
            a, b = directed
            boundary.append({
                "face": int(fi), "owner": int(owner[fi]), "a": int(a), "b": int(b),
                "ka": _key(qv[a]), "kb": _key(qv[b]),
            })
    by_position_edge = {}
    for row in boundary:
        token = tuple(sorted((row["ka"], row["kb"])))
        by_position_edge.setdefault(token, []).append(row)

    qualified = []
    max_drift = 0.0
    for rows in by_position_edge.values():
        # More or fewer than two authored boundary incidences is ambiguous and
        # cannot mint a mechanical weld.
        if len(rows) != 2:
            continue
        left, right = rows
        if left["owner"] == right["owner"]:
            continue
        reverse = left["ka"] == right["kb"] and left["kb"] == right["ka"]
        if not reverse:
            continue
        endpoint_pairs = ((left["a"], right["b"]), (left["b"], right["a"]))
        endpoint_drifts = []
        for qa, qb in endpoint_pairs:
            endpoint_drifts.append(_trajectory_drift(posed, members[qa], members[qb], scale))
        drift = max(endpoint_drifts)
        if drift > float(trajectory_tolerance_over_scale):
            continue
        for qa, qb in endpoint_pairs:
            union(qa, qb)
        max_drift = max(max_drift, drift)
        qualified.append({
            "face_ids": [left["face"], right["face"]],
            "owner_ids": sorted([left["owner"], right["owner"]]),
            "owner_quotient_edge_left": [left["a"], left["b"]],
            "owner_quotient_edge_right": [right["a"], right["b"]],
            "reverse_orientation": True,
            "maximum_endpoint_trajectory_drift_over_scale": float(drift),
        })

    roots = [find(i) for i in range(len(qv))]
    unique_roots = sorted(set(roots))
    root_to_m = {root: i for i, root in enumerate(unique_roots)}
    q_to_m = np.asarray([root_to_m[root] for root in roots], dtype=np.int32)
    raw_to_m = np.full(len(v), -1, dtype=np.int32)
    referenced = np.unique(f)
    raw_to_m[referenced] = q_to_m[raw_to_q[referenced]]
    mf = raw_to_m[f]
    if np.any(mf < 0):
        raise ValueError("MECHANICAL_SURFACE_FINAL_REMAP")
    mv = np.asarray([qv[root] for root in unique_roots], dtype=np.float64)

    raw_area = _double_area(v, f)
    mechanical_area = _double_area(mv, mf)
    collapsed = (mf[:, 0] == mf[:, 1]) | (mf[:, 1] == mf[:, 2]) | (mf[:, 2] == mf[:, 0])
    face_valid = (
        np.isfinite(raw_area) & (raw_area > 1.0e-12)
        & np.isfinite(mechanical_area) & (mechanical_area > 1.0e-12)
        & ~collapsed
    )

    component = _face_components(mf)
    owner_lists = []
    for mid in range(len(mv)):
        qids = np.flatnonzero(q_to_m == mid)
        owner_lists.append(sorted(set(map(int, qowner[qids].tolist()))))
    offsets = [0]
    values = []
    for row in owner_lists:
        values.extend(row); offsets.append(len(values))

    identity = hashlib.sha256(
        raw_to_m.tobytes(order="C")
        + mv.astype(np.float64, copy=False).tobytes(order="C")
        + mf.astype(np.int32, copy=False).tobytes(order="C")
        + owner.astype(np.int32, copy=False).tobytes(order="C")
        + face_valid.astype(np.bool_, copy=False).tobytes(order="C")
        + component.astype(np.int32, copy=False).tobytes(order="C")
        + np.asarray(offsets, np.int32).tobytes(order="C")
        + np.asarray(values, np.int32).tobytes(order="C")
    ).hexdigest()

    return MechanicalSurfaceTopologyAuthority(
        raw_to_mechanical_vertex=raw_to_m,
        mechanical_vertices_source_world=mv,
        mechanical_faces=mf.astype(np.int32),
        face_owner=owner.copy(),
        face_valid=face_valid.astype(np.bool_),
        surface_component_id=component.astype(np.int32),
        mechanical_vertex_owner_offsets=np.asarray(offsets, dtype=np.int32),
        mechanical_vertex_owner_values=np.asarray(values, dtype=np.int32),
        qualified_cross_owner_seam_count=len(qualified),
        qualified_cross_owner_seam_edges=tuple(qualified),
        maximum_qualified_seam_trajectory_drift_over_scale=float(max_drift),
        identity_sha256=identity,
    )
