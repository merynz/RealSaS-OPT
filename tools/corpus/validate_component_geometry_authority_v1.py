#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for RealSaS component-scoped geometry/sign authority.

Methodological separation from the producer is deliberate:
- no import from component_geometry_authority_v1 or the structural-audit helper;
- quotient/connectivity are reconstructed locally;
- topology partitions are compared label-invariantly;
- unsigned distance uses a direct point/triangle court;
- signed occupancy uses generalized solid-angle winding, not ray parity.

The court certifies candidate family evidence. It still does not decide final corpus
admission, split membership, or training readiness.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PARENT_SCHEMA = "RealSaS.ComponentGeometryAuthority.v1"
PARENT_STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.ComponentGeometryAuthorityCourt.v1"
STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bound(row, code):
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def matrix4(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1.0e-10, rtol=0)
        or np.linalg.matrix_rank(matrix[:3, :3]) != 3
    ):
        raise ValueError("COMPONENT_GEOMETRY_COURT_MATRIX")
    return matrix


def transform(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    return (homogeneous @ matrix.T)[:, :3].reshape(points.shape)


def position_key(value):
    value = np.asarray(value, dtype=np.float64).copy()
    value[value == 0.0] = 0.0
    return value.tobytes(order="C")


def independent_quotient(vertices, faces, owner):
    raw_owner = np.full(len(vertices), -1, dtype=np.int32)
    for fi, tri in enumerate(faces):
        oid = int(owner[fi])
        for raw in tri:
            raw = int(raw)
            prior = int(raw_owner[raw])
            if prior == -1:
                raw_owner[raw] = oid
            elif prior != oid:
                raise ValueError("COMPONENT_GEOMETRY_COURT_RAW_OWNER_CROSS")
    lookup = {}
    raw_to_q = np.full(len(vertices), -1, dtype=np.int32)
    qv = []
    qowner = []
    for raw in np.unique(faces):
        raw = int(raw)
        key = (int(raw_owner[raw]), position_key(vertices[raw]))
        if key not in lookup:
            lookup[key] = len(qv)
            qv.append(vertices[raw].copy())
            qowner.append(int(raw_owner[raw]))
        raw_to_q[raw] = lookup[key]
    return (
        raw_to_q,
        np.asarray(qv, dtype=np.float64),
        raw_to_q[faces].astype(np.int32),
        np.asarray(qowner, dtype=np.int32),
    )


def face_geometry(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(length) & (length > 1.0e-12)
    normal = np.zeros_like(cross)
    normal[valid] = cross[valid] / length[valid, None]
    return length, normal, valid


def edge_table(faces):
    table = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            table[(min(a, b), max(a, b))].append((fi, (a, b)))
    return dict(table)


def graph_components(face_count, graph, enabled=None):
    enabled = np.ones(face_count, dtype=np.bool_) if enabled is None else np.asarray(enabled, dtype=np.bool_)
    labels = np.full(face_count, -1, dtype=np.int32)
    cid = 0
    for start in range(face_count):
        if not enabled[start] or labels[start] >= 0:
            continue
        labels[start] = cid
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if enabled[neighbor] and labels[neighbor] < 0:
                    labels[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    return labels


def independent_surface_components(faces):
    graph = [set() for _ in range(len(faces))]
    for rows in edge_table(faces).values():
        ids = sorted({fi for fi, _ in rows})
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return graph_components(len(faces), graph)


def independent_sign_components(faces, face_valid):
    graph = [set() for _ in range(len(faces))]
    for rows in edge_table(faces).values():
        if len(rows) != 2:
            continue
        (left, d0), (right, d1) = rows
        if not (face_valid[left] and face_valid[right]):
            continue
        if d0[0] == d1[1] and d0[1] == d1[0]:
            graph[left].add(right)
            graph[right].add(left)
    return graph_components(len(faces), graph, enabled=face_valid)


def closed_vertex_links(faces):
    if not len(faces):
        return False
    incident = defaultdict(list)
    for fi, tri in enumerate(faces):
        for vertex in tri:
            incident[int(vertex)].append(fi)
    for vertex, face_ids in incident.items():
        graph = defaultdict(set)
        degree = defaultdict(int)
        for fi in face_ids:
            other = [int(value) for value in faces[fi] if int(value) != vertex]
            if len(other) != 2:
                return False
            a, b = other
            graph[a].add(b)
            graph[b].add(a)
            degree[a] += 1
            degree[b] += 1
        if not degree or any(value != 2 for value in degree.values()):
            return False
        seed = next(iter(degree))
        seen = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(seen) != len(degree):
            return False
    return True


def independent_component_validity(faces, labels, face_valid):
    table = edge_table(faces)
    result = {}
    for cid in sorted(set(int(x) for x in labels if int(x) >= 0)):
        face_ids = np.flatnonzero(labels == cid)
        face_set = set(map(int, face_ids))
        okay = bool(len(face_ids) and np.all(face_valid[face_ids]))
        for rows in table.values():
            local = [(fi, direction) for fi, direction in rows if fi in face_set]
            if not local:
                continue
            if len(local) != 2 or len(rows) != 2:
                okay = False
                continue
            d0, d1 = local[0][1], local[1][1]
            if not (d0[0] == d1[1] and d0[1] == d1[0]):
                okay = False
        if not closed_vertex_links(faces[face_ids]):
            okay = False
        result[cid] = okay
    return result


def partition_sets(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return {
        frozenset(map(int, np.flatnonzero(labels == cid)))
        for cid in sorted(set(map(int, labels.tolist())))
        if cid >= 0
    }


def compare_partition(expected, actual, code):
    if not np.array_equal(np.asarray(expected) < 0, np.asarray(actual) < 0):
        raise ValueError(code + "_UNASSIGNED_MASK")
    if partition_sets(expected) != partition_sets(actual):
        raise ValueError(code + "_PARTITION")


def component_valid_by_face_set(labels, validity):
    result = {}
    labels = np.asarray(labels, dtype=np.int64)
    validity = np.asarray(validity, dtype=np.bool_)
    for cid in sorted(set(map(int, labels.tolist()))):
        if cid < 0:
            continue
        if cid >= len(validity):
            raise ValueError("COMPONENT_GEOMETRY_COURT_VALIDITY_CARDINALITY")
        result[frozenset(map(int, np.flatnonzero(labels == cid)))] = bool(validity[cid])
    return result


def point_triangle_distance_sq(points, tri):
    p = points[:, None, :]
    a, b, c = tri[None, :, 0, :], tri[None, :, 1, :], tri[None, :, 2, :]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.sum(ab * ap, -1)
    d2 = np.sum(ac * ap, -1)
    bp = p - b
    d3 = np.sum(ab * bp, -1)
    d4 = np.sum(ac * bp, -1)
    cp = p - c
    d5 = np.sum(ab * cp, -1)
    d6 = np.sum(ac * cp, -1)
    dist = np.full(d1.shape, np.inf)
    mask = (d1 <= 0) & (d2 <= 0)
    dist[mask] = np.sum(ap * ap, -1)[mask]
    mask = (d3 >= 0) & (d4 <= d3)
    dist[mask] = np.minimum(dist[mask], np.sum(bp * bp, -1)[mask])
    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    den = d1 - d3
    t = np.divide(d1, den, out=np.zeros_like(d1), where=np.abs(den) > 1.0e-30)
    q = a + t[..., None] * ab
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, -1)[mask])
    mask = (d6 >= 0) & (d5 <= d6)
    dist[mask] = np.minimum(dist[mask], np.sum(cp * cp, -1)[mask])
    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    den = d2 - d6
    t = np.divide(d2, den, out=np.zeros_like(d2), where=np.abs(den) > 1.0e-30)
    q = a + t[..., None] * ac
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, -1)[mask])
    va = d3 * d6 - d5 * d4
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    edge = c - b
    den = (d4 - d3) + (d5 - d6)
    t = np.divide(d4 - d3, den, out=np.zeros_like(d4), where=np.abs(den) > 1.0e-30)
    q = b + t[..., None] * edge
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, -1)[mask])
    mask = ~np.isfinite(dist)
    den = va + vb + vc
    vv = np.divide(vb, den, out=np.zeros_like(vb), where=np.abs(den) > 1.0e-30)
    ww = np.divide(vc, den, out=np.zeros_like(vc), where=np.abs(den) > 1.0e-30)
    q = a + vv[..., None] * ab + ww[..., None] * ac
    fallback = np.sum((p - q) ** 2, -1)
    dist[mask] = fallback[mask]
    return dist


def direct_distance(points, vertices, faces, face_ids, face_chunk=1024):
    points = np.asarray(points, dtype=np.float64)
    best = np.full(len(points), np.inf, dtype=np.float64)
    tri_all = vertices[faces[face_ids]]
    for start in range(0, len(tri_all), face_chunk):
        distance_sq = point_triangle_distance_sq(points, tri_all[start : start + face_chunk])
        best = np.minimum(best, np.min(distance_sq, axis=1))
    return np.sqrt(best)


def solid_angle_inside(points, vertices, faces, chunk=16):
    tri = vertices[faces]
    result = []
    for start in range(0, len(points), chunk):
        p = np.asarray(points[start : start + chunk], dtype=np.float64)
        a = tri[None, :, 0, :] - p[:, None, :]
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la = np.linalg.norm(a, axis=-1)
        lb = np.linalg.norm(b, axis=-1)
        lc = np.linalg.norm(c, axis=-1)
        numerator = np.einsum("bij,bij->bi", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("bij,bij->bi", a, b) * lc
            + np.einsum("bij,bij->bi", b, c) * la
            + np.einsum("bij,bij->bi", c, a) * lb
        )
        angle = 2.0 * np.arctan2(numerator, denominator)
        winding = np.sum(angle, axis=1) / (4.0 * math.pi)
        result.append(np.abs(winding) > 0.5)
    return np.concatenate(result) if result else np.zeros(0, dtype=np.bool_)


def deterministic_subset(length, maximum, seed_text):
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    return np.sort(rng.choice(length, size=maximum, replace=False)).astype(np.int64)


def validate(manifest, *, maximum_surface_samples=128, maximum_distance_samples=48, maximum_sign_samples=96):
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("COMPONENT_GEOMETRY_COURT_PARENT")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("COMPONENT_GEOMETRY_COURT_SUBJECT")
    parents = dict(manifest.get("parents") or {})
    source_path = bound(parents.get("geometry_npz"), "COMPONENT_GEOMETRY_COURT_SOURCE_DRIFT")
    snapshot_path = bound(parents.get("source_scene_snapshot"), "COMPONENT_GEOMETRY_COURT_SNAPSHOT_DRIFT")
    bound(parents.get("camera_bundle"), "COMPONENT_GEOMETRY_COURT_CAMERA_DRIFT")
    outputs = dict(manifest.get("outputs") or {})
    topology_path = bound(outputs.get("topology"), "COMPONENT_GEOMETRY_COURT_TOPOLOGY_DRIFT")
    surface_path = bound(outputs.get("surface"), "COMPONENT_GEOMETRY_COURT_SURFACE_DRIFT")
    query_path = bound(outputs.get("distance_sign"), "COMPONENT_GEOMETRY_COURT_QUERY_DRIFT")

    with np.load(source_path, allow_pickle=False) as z:
        vertices_source = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        owner = np.asarray(z["face_owner_privileged"], dtype=np.int32)
    snapshot = json.loads(snapshot_path.read_text())
    matrix = matrix4(snapshot["source_world_to_canonical"])
    vertices_canonical = transform(vertices_source, matrix)
    raw_to_q, qsource, qfaces, qowner = independent_quotient(vertices_source, faces, owner)
    qcanonical = transform(qsource, matrix)
    raw_length, raw_normal, raw_valid = face_geometry(vertices_source, faces)
    can_length, can_normal, can_valid = face_geometry(vertices_canonical, faces)
    q_length, _, q_valid_area = face_geometry(qsource, qfaces)
    collapsed = (
        (qfaces[:, 0] == qfaces[:, 1])
        | (qfaces[:, 1] == qfaces[:, 2])
        | (qfaces[:, 2] == qfaces[:, 0])
    )
    face_valid = raw_valid & can_valid & q_valid_area & ~collapsed
    surface_labels = independent_surface_components(qfaces)
    sign_labels = independent_sign_components(qfaces, face_valid)
    sign_valid_map = independent_component_validity(qfaces, sign_labels, face_valid)

    with np.load(topology_path, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world", "rest_vertices_canonical", "faces",
            "face_owner_privileged", "face_normal_source_world", "face_normal_canonical",
            "face_valid", "raw_to_topology_vertex", "topology_vertices_source_world",
            "topology_vertices_canonical", "topology_faces", "topology_vertex_owner",
            "surface_component_id", "sign_component_id", "sign_component_valid",
        }
        if not required.issubset(z.files):
            raise ValueError("COMPONENT_GEOMETRY_COURT_TOPOLOGY_ARRAYS")
        exact = {
            "rest_vertices_source_world": vertices_source.astype(np.float32),
            "rest_vertices_canonical": vertices_canonical.astype(np.float32),
            "faces": faces.astype(np.int32),
            "face_owner_privileged": owner.astype(np.int32),
            "face_valid": face_valid.astype(np.bool_),
            "raw_to_topology_vertex": raw_to_q.astype(np.int32),
            "topology_vertices_source_world": qsource.astype(np.float32),
            "topology_vertices_canonical": qcanonical.astype(np.float32),
            "topology_faces": qfaces.astype(np.int32),
            "topology_vertex_owner": qowner.astype(np.int32),
        }
        for name, expected in exact.items():
            if not np.array_equal(np.asarray(z[name]), expected):
                raise ValueError("COMPONENT_GEOMETRY_COURT_TOPOLOGY_MISMATCH:" + name)
        if not np.allclose(np.asarray(z["face_normal_source_world"]), raw_normal.astype(np.float32), atol=2e-6, rtol=0):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SOURCE_NORMAL")
        if not np.allclose(np.asarray(z["face_normal_canonical"]), can_normal.astype(np.float32), atol=2e-6, rtol=0):
            raise ValueError("COMPONENT_GEOMETRY_COURT_CANONICAL_NORMAL")
        producer_surface_labels = np.asarray(z["surface_component_id"], dtype=np.int32)
        producer_sign_labels = np.asarray(z["sign_component_id"], dtype=np.int32)
        producer_sign_valid = np.asarray(z["sign_component_valid"], dtype=np.bool_)
    compare_partition(surface_labels, producer_surface_labels, "COMPONENT_GEOMETRY_COURT_SURFACE_COMPONENT")
    compare_partition(sign_labels, producer_sign_labels, "COMPONENT_GEOMETRY_COURT_SIGN_COMPONENT")
    expected_valid_by_set = {
        frozenset(map(int, np.flatnonzero(sign_labels == cid))): bool(value)
        for cid, value in sign_valid_map.items()
    }
    actual_valid_by_set = component_valid_by_face_set(producer_sign_labels, producer_sign_valid)
    if expected_valid_by_set != actual_valid_by_set:
        raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_COMPONENT_VALIDITY")

    with np.load(surface_path, allow_pickle=False) as z:
        required = {"triangle_id", "barycentric", "surface_xyz_canonical", "normal_canonical", "normal_valid"}
        if not required.issubset(z.files):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SURFACE_ARRAYS")
        triangle_id = np.asarray(z["triangle_id"], dtype=np.int64)
        bary = np.asarray(z["barycentric"], dtype=np.float64)
        surface_xyz = np.asarray(z["surface_xyz_canonical"], dtype=np.float64)
        surface_normal = np.asarray(z["normal_canonical"], dtype=np.float64)
        normal_valid = np.asarray(z["normal_valid"], dtype=np.bool_)
    sample = deterministic_subset(len(triangle_id), maximum_surface_samples, subject_id + "|surface-court")
    if np.any(triangle_id < 0) or np.any(triangle_id >= len(faces)) or np.any(~face_valid[triangle_id]):
        raise ValueError("COMPONENT_GEOMETRY_COURT_SURFACE_FACE")
    if bary.shape != (len(triangle_id), 3) or not np.allclose(bary.sum(axis=1), 1.0, atol=2e-6, rtol=0) or np.any(bary < -2e-6):
        raise ValueError("COMPONENT_GEOMETRY_COURT_BARYCENTRIC")
    expected_xyz = np.sum(vertices_canonical[faces[triangle_id[sample]]] * bary[sample, :, None], axis=1)
    if not np.allclose(surface_xyz[sample], expected_xyz, atol=3e-6, rtol=0):
        raise ValueError("COMPONENT_GEOMETRY_COURT_SURFACE_XYZ")
    if not np.allclose(surface_normal[sample], can_normal[triangle_id[sample]], atol=2e-6, rtol=0):
        raise ValueError("COMPONENT_GEOMETRY_COURT_SURFACE_NORMAL")
    if not np.all(normal_valid):
        raise ValueError("COMPONENT_GEOMETRY_COURT_SURFACE_NORMAL_VALID")

    with np.load(query_path, allow_pickle=False) as z:
        required = {
            "query_xyz_canonical", "unsigned_distance", "unsigned_distance_valid",
            "closest_face", "query_nearest_sign_component",
            "query_nearest_component_is_sign_candidate", "signed_query_index",
            "signed_distance_value", "occupancy_value", "signed_component_id",
        }
        if not required.issubset(z.files):
            raise ValueError("COMPONENT_GEOMETRY_COURT_QUERY_ARRAYS")
        query = np.asarray(z["query_xyz_canonical"], dtype=np.float64)
        unsigned = np.asarray(z["unsigned_distance"], dtype=np.float64)
        unsigned_valid = np.asarray(z["unsigned_distance_valid"], dtype=np.bool_)
        closest_face = np.asarray(z["closest_face"], dtype=np.int64)
        nearest_sign_component = np.asarray(z["query_nearest_sign_component"], dtype=np.int32)
        nearest_sign_candidate = np.asarray(z["query_nearest_component_is_sign_candidate"], dtype=np.bool_)
        signed_index = np.asarray(z["signed_query_index"], dtype=np.int64)
        signed_value = np.asarray(z["signed_distance_value"], dtype=np.float64)
        occupancy = np.asarray(z["occupancy_value"], dtype=np.bool_)
        signed_component = np.asarray(z["signed_component_id"], dtype=np.int32)
    if query.ndim != 2 or query.shape[1:] != (3,) or not np.isfinite(query).all():
        raise ValueError("COMPONENT_GEOMETRY_COURT_QUERY_LAYOUT")
    if unsigned.shape != (len(query),) or unsigned_valid.shape != (len(query),) or not np.all(unsigned_valid):
        raise ValueError("COMPONENT_GEOMETRY_COURT_UNSIGNED_LAYOUT")
    if closest_face.shape != (len(query),) or np.any(closest_face < 0) or np.any(closest_face >= len(faces)) or np.any(~face_valid[closest_face]):
        raise ValueError("COMPONENT_GEOMETRY_COURT_CLOSEST_FACE")
    expected_nearest_component = sign_labels[closest_face]
    compare_component_valid = np.zeros(len(query), dtype=np.bool_)
    for cid, valid in sign_valid_map.items():
        compare_component_valid[expected_nearest_component == cid] = bool(valid)
    compare_partition_mask = expected_nearest_component < 0
    if not np.array_equal(compare_partition_mask, nearest_sign_component < 0):
        raise ValueError("COMPONENT_GEOMETRY_COURT_QUERY_COMPONENT_MASK")
    # Map producer local component IDs to independent IDs through their face sets.
    producer_set_to_independent = {}
    for producer_cid in sorted(set(map(int, producer_sign_labels.tolist()))):
        if producer_cid < 0:
            continue
        face_set = frozenset(map(int, np.flatnonzero(producer_sign_labels == producer_cid)))
        matches = [cid for cid in sign_valid_map if frozenset(map(int, np.flatnonzero(sign_labels == cid))) == face_set]
        if len(matches) != 1:
            raise ValueError("COMPONENT_GEOMETRY_COURT_COMPONENT_MAPPING")
        producer_set_to_independent[producer_cid] = matches[0]
    mapped_nearest = np.full(len(query), -1, dtype=np.int32)
    for producer_cid, independent_cid in producer_set_to_independent.items():
        mapped_nearest[nearest_sign_component == producer_cid] = independent_cid
    if not np.array_equal(mapped_nearest, expected_nearest_component):
        raise ValueError("COMPONENT_GEOMETRY_COURT_QUERY_COMPONENT")
    if not np.array_equal(nearest_sign_candidate, compare_component_valid):
        raise ValueError("COMPONENT_GEOMETRY_COURT_QUERY_SIGN_CANDIDATE")

    distance_sample = deterministic_subset(len(query), maximum_distance_samples, subject_id + "|distance-court")
    valid_face_ids = np.flatnonzero(face_valid)
    direct = direct_distance(query[distance_sample], vertices_canonical, faces, valid_face_ids)
    if not np.allclose(unsigned[distance_sample], direct, atol=5e-5, rtol=2e-5):
        raise ValueError("COMPONENT_GEOMETRY_COURT_UNSIGNED_DISTANCE")

    if not (len(signed_index) == len(signed_value) == len(occupancy) == len(signed_component)):
        raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_COMPACT_CARDINALITY")
    if len(signed_index):
        if np.any(signed_index < 0) or np.any(signed_index >= len(query)) or len(np.unique(signed_index)) != len(signed_index):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_INDEX")
        if not np.all(np.diff(signed_index) > 0):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_INDEX_ORDER")
        if not np.all(nearest_sign_candidate[signed_index]):
            raise ValueError("COMPONENT_GEOMETRY_COURT_INVALID_SIGN_ROW_EMITTED")
        if not np.allclose(np.abs(signed_value), unsigned[signed_index], atol=2e-6, rtol=0):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_MAGNITUDE")
        mapped_signed_component = np.asarray(
            [producer_set_to_independent.get(int(cid), -1) for cid in signed_component], dtype=np.int32
        )
        if not np.array_equal(mapped_signed_component, expected_nearest_component[signed_index]):
            raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_COMPONENT_BINDING")
        sign_sample_local = deterministic_subset(len(signed_index), maximum_sign_samples, subject_id + "|sign-court")
        for independent_cid in sorted(set(map(int, mapped_signed_component[sign_sample_local].tolist()))):
            local_rows = sign_sample_local[mapped_signed_component[sign_sample_local] == independent_cid]
            face_ids = np.flatnonzero(sign_labels == independent_cid)
            inside = solid_angle_inside(query[signed_index[local_rows]], qcanonical, qfaces[face_ids])
            if not np.array_equal(inside, occupancy[local_rows]):
                raise ValueError("COMPONENT_GEOMETRY_COURT_SOLID_ANGLE_OCCUPANCY")
            expected_sign = np.where(inside, -1.0, 1.0)
            actual_sign = np.where(signed_value[local_rows] < 0, -1.0, 1.0)
            if not np.array_equal(expected_sign, actual_sign):
                raise ValueError("COMPONENT_GEOMETRY_COURT_SIGN_DIRECTION")

    face_fraction = float(np.count_nonzero(face_valid) / max(len(faces), 1))
    signed_fraction = float(len(signed_index) / max(len(query), 1))
    if face_fraction == 1.0:
        surface_candidate = "QUALIFIED_CORE_CANDIDATE"
        normal_candidate = "QUALIFIED_CORE_CANDIDATE"
    elif face_fraction > 0.0:
        surface_candidate = "QUALIFIED_MASKED_CANDIDATE"
        normal_candidate = "QUALIFIED_MASKED_CANDIDATE"
    else:
        surface_candidate = "UNAVAILABLE_CANDIDATE"
        normal_candidate = "UNAVAILABLE_CANDIDATE"
    if signed_fraction == 1.0:
        sign_candidate = "QUALIFIED_CORE_CANDIDATE"
    elif signed_fraction > 0.0:
        sign_candidate = "QUALIFIED_MASKED_CANDIDATE"
    else:
        sign_candidate = "UNAVAILABLE_CANDIDATE"
    topology_candidate = "QUALIFIED_CORE_CANDIDATE" if face_fraction == 1.0 else "QUALIFIED_MASKED_CANDIDATE"

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "parent_manifest": {
            "path": str(Path(manifest["__manifest_path"]).resolve()) if manifest.get("__manifest_path") else None,
            "sha256": manifest.get("__manifest_sha256"),
        },
        "court_methods": {
            "quotient": "INDEPENDENT_OWNER_PLUS_EXACT_POSITION_RECONSTRUCTION",
            "partition_comparison": "LABEL_INVARIANT_FACE_SET_PARTITION",
            "unsigned_distance": "DIRECT_POINT_TRIANGLE_DISTANCE",
            "signed_occupancy": "GENERALIZED_SOLID_ANGLE_WINDING",
            "producer_ray_parity_reused": False,
            "producer_topology_helper_imported": False,
        },
        "counts": {
            "face_count": int(len(faces)),
            "valid_face_count": int(np.count_nonzero(face_valid)),
            "surface_component_count": int(len(partition_sets(surface_labels))),
            "sign_candidate_component_count": int(len(partition_sets(sign_labels))),
            "sign_valid_component_count": int(sum(sign_valid_map.values())),
            "query_count": int(len(query)),
            "signed_query_count": int(len(signed_index)),
            "surface_sample_court_count": int(len(sample)),
            "distance_court_count": int(len(distance_sample)),
            "sign_court_count": int(min(len(signed_index), maximum_sign_samples)),
        },
        "coverage": {
            "surface_face_fraction": face_fraction,
            "signed_query_fraction": signed_fraction,
        },
        "family_candidate_outcomes": {
            "SURFACE_GEOMETRY": surface_candidate,
            "SURFACE_TOPOLOGY": topology_candidate,
            "SIGNED_DISTANCE_OCCUPANCY": sign_candidate,
            "ORIENTED_NORMALS": normal_candidate,
        },
        "authority": {
            "independent_court_passed": True,
            "final_subject_disposition_minted": False,
            "training_authorized": False,
            "split_membership_minted": False,
        },
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.geometry_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest["__manifest_path"] = str(manifest_path)
    manifest["__manifest_sha256"] = sha256(manifest_path)
    payload = validate(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "subject_id": payload["subject_id"],
        "family_candidate_outcomes": payload["family_candidate_outcomes"],
        "court_identity_sha256": payload["court_identity_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
