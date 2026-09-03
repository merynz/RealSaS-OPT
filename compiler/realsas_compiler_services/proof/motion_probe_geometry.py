from __future__ import annotations

"""Deterministic geometry primitives for current V4 authored-motion proof.

This module is subordinate measurement code. 4x4 matrices exist only as an
internal LBS evaluation carrier for puppet-local 2D/2.5D authored transforms.
They are not a full-3D product or motion authority.
"""

import math
from types import SimpleNamespace
import numpy as np

from compiler.realsas_compiler_services.numerics.lbs import apply_lbs_probe_v1


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (float(x), float(y), float(z))
    return out


def authored_delta_matrix(key) -> np.ndarray:
    tx, ty = map(float, key.translation_xy)
    sx, sy = map(float, key.scale_xy)
    rz = math.radians(float(key.rotation_deg))
    dz = float(key.depth_offset)
    if not all(math.isfinite(v) for v in (tx, ty, sx, sy, rz, dz)):
        raise ValueError("non-finite authored motion key")
    c, s = math.cos(rz), math.sin(rz)
    out = np.eye(4, dtype=np.float64)
    out[0, 0], out[0, 1] = c * sx, -s * sy
    out[1, 0], out[1, 1] = s * sx, c * sy
    out[0, 3], out[1, 3], out[2, 3] = tx, ty, dz
    return out


def identity_key(time_sec: float):
    return SimpleNamespace(
        time_sec=float(time_sec), translation_xy=(0.0, 0.0), rotation_deg=0.0,
        scale_xy=(1.0, 1.0), depth_offset=0.0,
    )


def interpolate_key(keys, time_sec: float):
    if not keys:
        return identity_key(time_sec)
    ordered = tuple(sorted(keys, key=lambda k: float(k.time_sec)))
    if time_sec <= float(ordered[0].time_sec):
        return ordered[0]
    if time_sec >= float(ordered[-1].time_sec):
        return ordered[-1]
    for a, b in zip(ordered[:-1], ordered[1:]):
        ta, tb = float(a.time_sec), float(b.time_sec)
        if ta <= time_sec <= tb:
            if tb <= ta:
                return b
            u = (time_sec - ta) / (tb - ta)
            lerp = lambda x, y: (1.0 - u) * float(x) + u * float(y)
            return SimpleNamespace(
                time_sec=float(time_sec),
                translation_xy=tuple(lerp(x, y) for x, y in zip(a.translation_xy, b.translation_xy)),
                rotation_deg=lerp(a.rotation_deg, b.rotation_deg),
                scale_xy=tuple(lerp(x, y) for x, y in zip(a.scale_xy, b.scale_xy)),
                depth_offset=lerp(a.depth_offset, b.depth_offset),
            )
    return ordered[-1]


def joint_topological_order(skeleton):
    joints = tuple(sorted(skeleton.joints, key=lambda j: j.canonical_joint_id))
    if not joints:
        raise ValueError("motion probe requires qualified joints")
    by_id = {j.canonical_joint_id: j for j in joints}
    if len(by_id) != len(joints):
        raise ValueError("duplicate canonical joint id")
    for joint in joints:
        parent = joint.parent_canonical_id
        if parent is not None and parent not in by_id:
            raise ValueError("qualified skeleton contains missing parent")
    visiting, done, ordered = set(), set(), []
    def visit(jid):
        if jid in done:
            return
        if jid in visiting:
            raise ValueError("qualified skeleton contains cycle")
        visiting.add(jid)
        parent = by_id[jid].parent_canonical_id
        if parent is not None:
            visit(parent)
        visiting.remove(jid); done.add(jid); ordered.append(jid)
    for jid in sorted(by_id):
        visit(jid)
    return tuple(ordered), by_id


def clip_tracks(motion_state, clip_id: str) -> dict[str, object]:
    out = {}
    for track in motion_state.joint_tracks:
        if track.clip_id != clip_id:
            continue
        if track.canonical_joint_id in out:
            raise ValueError("duplicate joint track for clip")
        out[track.canonical_joint_id] = track
    return out


def sample_times(clip, tracks: dict[str, object], uniform_count: int) -> tuple[float, ...]:
    duration = float(clip.duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("motion clip duration must be finite positive")
    times = {float(x) for x in np.linspace(0.0, duration, int(uniform_count))}
    for track in tracks.values():
        for key in track.keys:
            t = float(key.time_sec)
            if not math.isfinite(t) or t < -1e-9 or t > duration + 1e-9:
                raise ValueError("motion key outside clip duration")
            times.add(min(duration, max(0.0, t)))
    return tuple(sorted(times))


def skinning_transforms(skeleton, tracks, time_sec: float, order, by_id) -> np.ndarray:
    posed, bind = {}, {}
    for jid in order:
        joint = by_id[jid]
        p = np.asarray(joint.position, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("invalid qualified joint position")
        parent = joint.parent_canonical_id
        key = interpolate_key(tracks[jid].keys, time_sec) if jid in tracks else identity_key(time_sec)
        delta = authored_delta_matrix(key)
        bind[jid] = translation_matrix(*p)
        if parent is None:
            posed[jid] = translation_matrix(*p) @ delta
        else:
            pp = np.asarray(by_id[parent].position, dtype=np.float64)
            posed[jid] = posed[parent] @ translation_matrix(*(p - pp)) @ delta
    return np.asarray([posed[jid] @ np.linalg.inv(bind[jid]) for jid in sorted(by_id)], dtype=np.float64)


def mesh_arrays(mesh, mesh_skin, skeleton):
    joint_ids = tuple(sorted(j.canonical_joint_id for j in skeleton.joints))
    ji = {jid: i for i, jid in enumerate(joint_ids)}
    vertices = tuple(mesh.vertices)
    vi = {v.canonical_mesh_vertex_id: i for i, v in enumerate(vertices)}
    if len(vi) != len(vertices):
        raise ValueError("duplicate qualified mesh vertex id")
    points = np.asarray([v.P for v in vertices], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("qualified mesh points invalid")
    rows = {r.canonical_mesh_vertex_id: r for r in mesh_skin.rows}
    weights = np.zeros((len(vertices), len(joint_ids)), dtype=np.float64)
    for i, vertex in enumerate(vertices):
        row = rows.get(vertex.canonical_mesh_vertex_id)
        if row is None:
            raise ValueError("motion probe missing mesh skin row")
        for jid, value in row.influences:
            if jid not in ji:
                raise ValueError("mesh skin references unknown canonical joint")
            weights[i, ji[jid]] = float(value)
    if not np.isfinite(weights).all() or (weights < -1e-8).any() or not np.allclose(weights.sum(1), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("motion probe requires qualified simplex mesh skin")
    faces = []
    for face in mesh.faces:
        if len(face) != 3 or any(vertex_id not in vi for vertex_id in face):
            raise ValueError("motion probe requires triangulated qualified mesh")
        faces.append(tuple(vi[vertex_id] for vertex_id in face))
    if not faces:
        raise ValueError("motion probe requires non-empty mesh faces")
    return points, weights, tuple(faces)


def bbox_diag(points: np.ndarray, eps: float) -> float:
    return max(float(np.linalg.norm(points.max(0) - points.min(0))) if len(points) else 1.0, float(eps))


def triangle_areas(points: np.ndarray, faces) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


def component_measurements(rest, deformed, faces, *, geometry_epsilon: float) -> dict:
    if deformed.ndim != 3 or deformed.shape[1:] != rest.shape:
        raise ValueError("deformed motion probe shape mismatch")
    nonfinite = int((~np.isfinite(deformed)).sum())
    if nonfinite:
        return {"nonfinite_value_count": nonfinite, "max_edge_relative_change": float("inf"),
                "p95_edge_relative_change": float("inf"), "min_triangle_area_ratio": 0.0,
                "max_triangle_area_ratio": float("inf"), "degenerate_triangle_instances": len(faces) * len(deformed),
                "max_motion_normalized": float("inf")}
    edges = sorted({tuple(sorted(pair)) for a, b, c in faces for pair in ((a, b), (b, c), (c, a))})
    e = np.asarray(edges, dtype=np.int64)
    rest_edge = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=-1)
    valid_edge = rest_edge > geometry_epsilon
    changes = []
    for frame in deformed:
        cur = np.linalg.norm(frame[e[:, 0]] - frame[e[:, 1]], axis=-1)
        if valid_edge.any():
            changes.append(np.abs(cur[valid_edge] / rest_edge[valid_edge] - 1.0))
    changes = np.concatenate(changes) if changes else np.zeros(0, dtype=np.float64)
    rest_area = triangle_areas(rest, faces); valid_face = rest_area > geometry_epsilon
    ratios, degenerate = [], 0
    for frame in deformed:
        area = triangle_areas(frame, faces)
        degenerate += int((area <= geometry_epsilon).sum())
        if valid_face.any():
            ratios.append(area[valid_face] / rest_area[valid_face])
    ratios = np.concatenate(ratios) if ratios else np.ones(1, dtype=np.float64)
    displacement = np.linalg.norm(deformed - rest[None], axis=-1) / bbox_diag(rest, geometry_epsilon)
    return {
        "nonfinite_value_count": 0,
        "max_edge_relative_change": float(changes.max()) if changes.size else 0.0,
        "p95_edge_relative_change": float(np.percentile(changes, 95)) if changes.size else 0.0,
        "min_triangle_area_ratio": float(ratios.min()), "max_triangle_area_ratio": float(ratios.max()),
        "degenerate_triangle_instances": int(degenerate),
        "max_motion_normalized": float(displacement.max()) if displacement.size else 0.0,
    }


def deform_component(rest, weights, transforms):
    return apply_lbs_probe_v1(rest, weights, transforms)
