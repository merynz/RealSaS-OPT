from __future__ import annotations

"""Cheap structural geometry audit for RealSaS corpus candidates.

This module does not repair source geometry and does not mint sign/SDF authority.
It measures the authored surface under an owner-preserving exact-position quotient
and records evidence needed to decide whether a stronger cross-owner equivalence
relation is justified.

Key invariants:
- raw/render vertex identity is preserved and never treated as topology identity;
- owner boundaries are preserved for the baseline quotient;
- face components are edge-connected, never merely vertex-connected;
- exact cross-owner coincidence is evidence only, not an automatic weld;
- posed trajectory compatibility is measured when available;
- all outputs are diagnostic/candidate authority, not training admission.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


SCHEMA = "RealSaS.CorpusStructuralGeometryAudit.v1"


def _mesh(vertices, faces, face_owner):
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
        raise ValueError("STRUCTURAL_AUDIT_FINITE_TRIANGULATED_MESH_REQUIRED")
    return v, f, owner


def _position_key(position: np.ndarray) -> bytes:
    p = np.asarray(position, dtype=np.float64).copy()
    p[p == 0.0] = 0.0
    return p.tobytes(order="C")


def _raw_vertex_owner(vertex_count: int, faces: np.ndarray, face_owner: np.ndarray) -> np.ndarray:
    result = np.full(vertex_count, -1, dtype=np.int32)
    for fi, tri in enumerate(faces):
        oid = int(face_owner[fi])
        for raw in tri:
            raw = int(raw)
            previous = int(result[raw])
            if previous == -1:
                result[raw] = oid
            elif previous != oid:
                raise ValueError("STRUCTURAL_AUDIT_RAW_VERTEX_CROSSES_FACE_OWNERS")
    return result


@dataclass(frozen=True)
class ExactOwnerQuotient:
    raw_to_topology: np.ndarray
    topology_vertices: np.ndarray
    topology_faces: np.ndarray
    topology_owner: np.ndarray
    raw_members: tuple[tuple[int, ...], ...]


def owner_preserving_exact_quotient(vertices, faces, face_owner) -> ExactOwnerQuotient:
    v, f, owner = _mesh(vertices, faces, face_owner)
    raw_owner = _raw_vertex_owner(len(v), f, owner)
    raw_to_q = np.full(len(v), -1, dtype=np.int32)
    qv: list[np.ndarray] = []
    qowner: list[int] = []
    members: list[list[int]] = []
    lookup: dict[tuple[int, bytes], int] = {}

    for raw in np.unique(f):
        raw = int(raw)
        oid = int(raw_owner[raw])
        if oid < 0:
            raise ValueError("STRUCTURAL_AUDIT_REFERENCED_VERTEX_OWNER_MISSING")
        key = (oid, _position_key(v[raw]))
        qid = lookup.get(key)
        if qid is None:
            qid = len(qv)
            lookup[key] = qid
            qv.append(v[raw].copy())
            qowner.append(oid)
            members.append([])
        raw_to_q[raw] = qid
        members[qid].append(raw)

    qv_array = np.asarray(qv, dtype=np.float64)
    qf = raw_to_q[f]
    if np.any(qf < 0):
        raise ValueError("STRUCTURAL_AUDIT_QUOTIENT_REMAP_FAILED")
    return ExactOwnerQuotient(
        raw_to_topology=raw_to_q,
        topology_vertices=qv_array,
        topology_faces=qf.astype(np.int32),
        topology_owner=np.asarray(qowner, dtype=np.int32),
        raw_members=tuple(tuple(row) for row in members),
    )


def _face_double_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )


def _edge_table(faces: np.ndarray):
    edge_faces: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append((fi, (a, b)))
    return edge_faces


def edge_connected_face_components(faces: np.ndarray) -> np.ndarray:
    """Return face-component IDs joined only through shared full edges."""
    f = np.asarray(faces, dtype=np.int64)
    graph = [set() for _ in range(len(f))]
    for rows in _edge_table(f).values():
        ids = [fi for fi, _ in rows]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                graph[a].add(b)
                graph[b].add(a)

    component = np.full(len(f), -1, dtype=np.int32)
    cid = 0
    for start in range(len(f)):
        if component[start] >= 0:
            continue
        component[start] = cid
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if component[neighbor] < 0:
                    component[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    return component


def vertex_connected_face_components(faces: np.ndarray) -> np.ndarray:
    """Diagnostic only: intentionally weaker than the canonical edge relation."""
    f = np.asarray(faces, dtype=np.int64)
    vertex_faces: dict[int, list[int]] = {}
    for fi, tri in enumerate(f):
        for vertex in tri:
            vertex_faces.setdefault(int(vertex), []).append(fi)
    graph = [set() for _ in range(len(f))]
    for ids in vertex_faces.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                graph[a].add(b)
                graph[b].add(a)
    component = np.full(len(f), -1, dtype=np.int32)
    cid = 0
    for start in range(len(f)):
        if component[start] >= 0:
            continue
        component[start] = cid
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if component[neighbor] < 0:
                    component[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    return component


def _edge_diagnostics(faces: np.ndarray):
    table = _edge_table(faces)
    boundary = []
    nonmanifold = []
    orientation_conflict = []
    for edge, rows in table.items():
        if len(rows) == 1:
            boundary.append((edge, rows[0]))
        elif len(rows) > 2:
            nonmanifold.append((edge, rows))
        elif len(rows) == 2:
            (_, d0), (_, d1) = rows
            if not (d0[0] == d1[1] and d0[1] == d1[0]):
                orientation_conflict.append((edge, rows))
    return table, boundary, nonmanifold, orientation_conflict


def _component_diagnostics(
    faces: np.ndarray,
    face_component: np.ndarray,
    face_valid: np.ndarray,
):
    table = _edge_table(faces)
    out = []
    for cid in sorted(set(map(int, face_component.tolist()))):
        ids = np.flatnonzero(face_component == cid)
        id_set = set(map(int, ids))
        boundary = 0
        nonmanifold = 0
        orientation_conflict = 0
        for _, rows in table.items():
            local = [(fi, direction) for fi, direction in rows if fi in id_set]
            if not local:
                continue
            if len(local) == 1:
                boundary += 1
            elif len(local) > 2:
                nonmanifold += 1
            elif len(local) == 2:
                d0, d1 = local[0][1], local[1][1]
                if not (d0[0] == d1[1] and d0[1] == d1[0]):
                    orientation_conflict += 1
        all_faces_valid = bool(np.all(face_valid[ids]))
        closed_candidate = bool(
            boundary == 0
            and nonmanifold == 0
            and orientation_conflict == 0
            and all_faces_valid
        )
        out.append(
            {
                "component_id": cid,
                "face_count": int(len(ids)),
                "boundary_edge_count": int(boundary),
                "nonmanifold_edge_count": int(nonmanifold),
                "orientation_conflict_edge_count": int(orientation_conflict),
                "all_faces_nondegenerate": all_faces_valid,
                "owner_preserving_closed_sign_candidate": closed_candidate,
            }
        )
    return out


def _trajectory_array(posed_vertices, vertex_count: int):
    if posed_vertices is None:
        return None
    posed = np.asarray(posed_vertices, dtype=np.float64)
    if posed.ndim == 2 and posed.shape == (vertex_count, 3):
        posed = posed[None, ...]
    if (
        posed.ndim != 3
        or posed.shape[1:] != (vertex_count, 3)
        or not np.isfinite(posed).all()
    ):
        raise ValueError("STRUCTURAL_AUDIT_POSED_VERTEX_LAYOUT")
    return posed


def _trajectory_group_drift(posed: np.ndarray | None, members: Iterable[int], scale: float):
    members = list(map(int, members))
    if posed is None or len(members) < 2:
        return None
    p = posed[:, members, :]
    max_abs = 0.0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            max_abs = max(max_abs, float(np.max(np.linalg.norm(p[:, i] - p[:, j], axis=-1))))
    return {
        "max_abs_source_units": max_abs,
        "max_over_subject_scale": max_abs / max(scale, 1e-12),
    }


def _position_groups(vertices: np.ndarray, faces: np.ndarray, raw_owner: np.ndarray):
    groups: dict[bytes, list[int]] = {}
    for raw in np.unique(faces):
        raw = int(raw)
        groups.setdefault(_position_key(vertices[raw]), []).append(raw)
    return groups


def _cross_owner_position_evidence(
    vertices: np.ndarray,
    faces: np.ndarray,
    raw_owner: np.ndarray,
    posed: np.ndarray | None,
    scale: float,
):
    groups = _position_groups(vertices, faces, raw_owner)
    rows = []
    for members in groups.values():
        owners = sorted(set(int(raw_owner[raw]) for raw in members))
        if len(owners) < 2:
            continue
        drift = _trajectory_group_drift(posed, members, scale)
        rows.append(
            {
                "owner_ids": owners,
                "raw_vertex_count": int(len(members)),
                "position": vertices[members[0]].astype(float).tolist(),
                "trajectory": drift,
            }
        )
    return rows


def _cross_owner_boundary_pairs(
    quotient: ExactOwnerQuotient,
    face_owner: np.ndarray,
    posed: np.ndarray | None,
    scale: float,
):
    _, boundary, _, _ = _edge_diagnostics(quotient.topology_faces)
    by_position_edge: dict[tuple[bytes, bytes], list[dict]] = {}
    for (a, b), (fi, directed) in boundary:
        pa = _position_key(quotient.topology_vertices[a])
        pb = _position_key(quotient.topology_vertices[b])
        key = tuple(sorted((pa, pb)))
        by_position_edge.setdefault(key, []).append(
            {
                "owner": int(face_owner[fi]),
                "face": int(fi),
                "edge": (int(a), int(b)),
                "directed": tuple(map(int, directed)),
            }
        )

    owner_pairs: dict[tuple[int, int], dict] = {}
    matched_edges = 0
    orientation_conflicts = 0
    trajectory_incompatible = 0
    trajectory_unknown = 0

    for rows in by_position_edge.values():
        owners = sorted(set(row["owner"] for row in rows))
        if len(owners) < 2:
            continue
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                if left["owner"] == right["owner"]:
                    continue
                pair = tuple(sorted((left["owner"], right["owner"])))
                summary = owner_pairs.setdefault(
                    pair,
                    {
                        "owner_ids": list(pair),
                        "matched_boundary_edge_count": 0,
                        "reverse_orientation_count": 0,
                        "same_orientation_count": 0,
                        "trajectory_compatible_count": 0,
                        "trajectory_incompatible_count": 0,
                        "trajectory_unknown_count": 0,
                        "max_trajectory_drift_over_scale": 0.0,
                    },
                )
                summary["matched_boundary_edge_count"] += 1
                matched_edges += 1

                l0, l1 = left["directed"]
                r0, r1 = right["directed"]
                lp0 = _position_key(quotient.topology_vertices[l0])
                lp1 = _position_key(quotient.topology_vertices[l1])
                rp0 = _position_key(quotient.topology_vertices[r0])
                rp1 = _position_key(quotient.topology_vertices[r1])
                reverse = lp0 == rp1 and lp1 == rp0
                if reverse:
                    summary["reverse_orientation_count"] += 1
                else:
                    summary["same_orientation_count"] += 1
                    orientation_conflicts += 1

                endpoint_drifts = []
                all_qids = left["edge"] + right["edge"]
                for position_key in (lp0, lp1):
                    raw_members = []
                    for qid in all_qids:
                        if _position_key(quotient.topology_vertices[qid]) == position_key:
                            raw_members.extend(quotient.raw_members[qid])
                    drift = _trajectory_group_drift(
                        posed, sorted(set(raw_members)), scale
                    )
                    if drift is not None:
                        endpoint_drifts.append(float(drift["max_over_subject_scale"]))
                if not endpoint_drifts:
                    summary["trajectory_unknown_count"] += 1
                    trajectory_unknown += 1
                else:
                    value = max(endpoint_drifts)
                    summary["max_trajectory_drift_over_scale"] = max(
                        summary["max_trajectory_drift_over_scale"], value
                    )
                    if value <= 1.0e-9:
                        summary["trajectory_compatible_count"] += 1
                    else:
                        summary["trajectory_incompatible_count"] += 1
                        trajectory_incompatible += 1

    return {
        "matched_boundary_edge_count": int(matched_edges),
        "orientation_conflict_count": int(orientation_conflicts),
        "trajectory_incompatible_count": int(trajectory_incompatible),
        "trajectory_unknown_count": int(trajectory_unknown),
        "owner_pairs": [owner_pairs[key] for key in sorted(owner_pairs)],
    }


def audit_structural_geometry_v1(
    vertices,
    faces,
    face_owner,
    *,
    posed_vertices=None,
    subject_id: str | None = None,
):
    v, f, owner = _mesh(vertices, faces, face_owner)
    posed = _trajectory_array(posed_vertices, len(v))
    raw_owner = _raw_vertex_owner(len(v), f, owner)
    q = owner_preserving_exact_quotient(v, f, owner)

    raw_area2 = _face_double_area(v, f)
    q_collapsed = (
        (q.topology_faces[:, 0] == q.topology_faces[:, 1])
        | (q.topology_faces[:, 1] == q.topology_faces[:, 2])
        | (q.topology_faces[:, 2] == q.topology_faces[:, 0])
    )
    q_area2 = _face_double_area(q.topology_vertices, q.topology_faces)
    raw_valid = np.isfinite(raw_area2) & (raw_area2 > 1.0e-12)
    q_valid = (~q_collapsed) & np.isfinite(q_area2) & (q_area2 > 1.0e-12)

    edge_component = edge_connected_face_components(q.topology_faces)
    vertex_component = vertex_connected_face_components(q.topology_faces)
    _, boundary, nonmanifold, orientation_conflict = _edge_diagnostics(q.topology_faces)
    components = _component_diagnostics(q.topology_faces, edge_component, q_valid)

    scale = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    cross_position = _cross_owner_position_evidence(v, f, raw_owner, posed, scale)
    cross_boundary = _cross_owner_boundary_pairs(q, owner, posed, scale)

    point_only_contact = bool(
        len(set(map(int, edge_component))) > len(set(map(int, vertex_component)))
    )

    classes = []
    referenced_raw = np.unique(f)
    collapse_count = int(len(referenced_raw) - len(q.topology_vertices))
    if collapse_count:
        classes.append("WITHIN_OWNER_EXACT_RENDER_SEAM_SPLIT")
    if len(set(map(int, owner.tolist()))) > 1:
        classes.append("MULTI_OWNER_AUTHORED_SCENE")
    if cross_position:
        classes.append("CROSS_OWNER_EXACT_POSITION_COINCIDENCE")
    if cross_boundary["matched_boundary_edge_count"]:
        classes.append("CROSS_OWNER_BOUNDARY_SEAM_CANDIDATE")
    if boundary:
        classes.append("OPEN_BOUNDARY_PRESENT_UNDER_OWNER_PRESERVING_QUOTIENT")
    if nonmanifold:
        classes.append("NONMANIFOLD_EDGE_PRESENT")
    if orientation_conflict:
        classes.append("ORIENTATION_CONFLICT_PRESENT")
    if np.count_nonzero(~raw_valid) or np.count_nonzero(q_collapsed) or np.count_nonzero(~q_valid):
        classes.append("DEGENERATE_FACE_EVIDENCE")
    if point_only_contact:
        classes.append("POINT_ONLY_CONTACT_PRESENT")

    payload = {
        "schema": SCHEMA,
        "subject_id": str(subject_id or ""),
        "status": "PASS_STRUCTURAL_MEASUREMENT__NO_TRAINING_AUTHORITY_MINTED",
        "authority": {
            "source_mutated": False,
            "training_authorized": False,
            "sign_authority_minted": False,
            "cross_owner_weld_performed": False,
            "component_semantics": "EDGE_CONNECTED_FACE_COMPONENTS",
            "baseline_quotient": "SAME_FACE_OWNER_PLUS_EXACT_POSITION_ONLY__DIAGNOSTIC_BASELINE",
        },
        "counts": {
            "raw_vertex_count": int(len(v)),
            "referenced_raw_vertex_count": int(len(referenced_raw)),
            "face_count": int(len(f)),
            "owner_count": int(len(set(map(int, owner.tolist())))),
            "owner_preserving_topology_vertex_count": int(len(q.topology_vertices)),
            "within_owner_exact_duplicate_collapse_count": collapse_count,
            "edge_connected_face_component_count": int(len(set(map(int, edge_component.tolist())))),
            "vertex_connected_face_component_count_diagnostic": int(
                len(set(map(int, vertex_component.tolist())))
            ),
            "boundary_edge_count": int(len(boundary)),
            "nonmanifold_edge_count": int(len(nonmanifold)),
            "orientation_conflict_edge_count": int(len(orientation_conflict)),
            "raw_zero_area_face_count": int(np.count_nonzero(~raw_valid)),
            "quotient_collapsed_face_count": int(np.count_nonzero(q_collapsed)),
            "quotient_zero_area_face_count": int(np.count_nonzero(~q_valid)),
            "cross_owner_exact_position_group_count": int(len(cross_position)),
            "cross_owner_matched_boundary_edge_count": int(
                cross_boundary["matched_boundary_edge_count"]
            ),
        },
        "subject_scale": scale,
        "failure_classes_observed": classes,
        "edge_components": components,
        "cross_owner_position_groups": cross_position,
        "cross_owner_boundary_evidence": cross_boundary,
        "diagnostics": {
            "point_only_contact_detected": point_only_contact,
            "posed_trajectory_evidence_available": posed is not None,
            "nested_shell_status": "NOT_ASSESSED_BY_CHEAP_STRUCTURAL_AUDIT",
            "self_intersection_status": "NOT_ASSESSED_BY_CHEAP_STRUCTURAL_AUDIT",
        },
    }
    identity_input = dict(payload)
    payload["audit_identity_sha256"] = hashlib.sha256(
        json.dumps(identity_input, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload
