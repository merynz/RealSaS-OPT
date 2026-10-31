#!/usr/bin/env python3
"""Independent court for Truth139 V3.2 geometric-topology/SDF V2.

This court deliberately does not import the V2 producer or its geometric-topology
helper. It independently reconstructs owner-aware exact-position topology,
recomputes source/canonical face normals and sampled surface addresses, verifies
unsigned distances, and validates signed occupancy with a solid-angle winding
court rather than producer ray parity.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

PARENT_SCHEMA = "RealSaS.Truth139V32ExactGeometry.v2"
PARENT_STATUS = "PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.Truth139V32ExactGeometryCourt.v2"
PASS_STATUS = "PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2_INDEPENDENT_COURT"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound(ref, code):
    path = Path(str((ref or {}).get("path") or "")).resolve()
    expected = str((ref or {}).get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def matrix4(value):
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or np.linalg.matrix_rank(m[:3, :3]) != 3:
        raise ValueError("V32_GEOMETRY_V2_COURT_MATRIX")
    return m


def transform(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    flat = p.reshape(-1, 3)
    h = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    return (h @ matrix.T)[:, :3].reshape(p.shape)


def independent_quotient(vertices, faces, face_owner):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    owner = np.asarray(face_owner, dtype=np.int32)
    if owner.shape != (len(f),):
        raise ValueError("V32_GEOMETRY_V2_COURT_OWNER_LAYOUT")
    raw_owner = np.full(len(v), -1, dtype=np.int32)
    for fi, tri in enumerate(f):
        oid = int(owner[fi])
        for raw in tri:
            raw = int(raw)
            if raw_owner[raw] == -1:
                raw_owner[raw] = oid
            elif raw_owner[raw] != oid:
                raise ValueError("V32_GEOMETRY_V2_COURT_RAW_VERTEX_OWNER_CROSS")
    lookup = {}
    raw_to_q = np.full(len(v), -1, dtype=np.int32)
    qv = []
    qowner = []
    for raw in np.unique(f):
        raw = int(raw)
        p = v[raw].copy()
        p[p == 0.0] = 0.0
        key = (int(raw_owner[raw]), p.tobytes(order="C"))
        if key not in lookup:
            lookup[key] = len(qv)
            qv.append(v[raw].copy())
            qowner.append(int(raw_owner[raw]))
        raw_to_q[raw] = lookup[key]
    qv = np.asarray(qv, dtype=np.float64)
    qf = raw_to_q[f]
    qowner = np.asarray(qowner, dtype=np.int32)
    if np.any(qf < 0):
        raise ValueError("V32_GEOMETRY_V2_COURT_QUOTIENT_REMAP")
    return raw_to_q, qv, qf.astype(np.int32), qowner


def independent_topology(vertex_count, faces):
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    edges, inverse, incidence = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    orient = np.ones(len(edges), dtype=np.bool_)
    for eid in np.flatnonzero(incidence == 2):
        rows = np.flatnonzero(inverse == eid)
        a, b = directed[rows[0]], directed[rows[1]]
        orient[eid] = bool(a[0] == b[1] and a[1] == b[0])
    edge_faces = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
    face_graph = [set() for _ in range(len(faces))]
    for rows in edge_faces.values():
        if len(rows) == 2:
            a, b = rows
            face_graph[a].add(b)
            face_graph[b].add(a)
    fc = np.full(len(faces), -1, dtype=np.int32)
    cid = 0
    for start in range(len(faces)):
        if fc[start] >= 0:
            continue
        stack = [start]
        fc[start] = cid
        while stack:
            current = stack.pop()
            for neighbor in face_graph[current]:
                if fc[neighbor] < 0:
                    fc[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    incident = [[] for _ in range(vertex_count)]
    for fi, tri in enumerate(faces):
        for vertex in tri:
            incident[int(vertex)].append(fi)
    link_valid = np.ones(vertex_count, dtype=np.bool_)
    for vertex, face_ids in enumerate(incident):
        if not face_ids:
            continue
        graph = defaultdict(set)
        degree = defaultdict(int)
        for fi in face_ids:
            other = [int(x) for x in faces[fi] if int(x) != vertex]
            if len(other) != 2:
                link_valid[vertex] = False
                break
            a, b = other
            graph[a].add(b)
            graph[b].add(a)
            degree[a] += 1
            degree[b] += 1
        if not link_valid[vertex]:
            continue
        if not degree or any(value != 2 for value in degree.values()):
            link_valid[vertex] = False
            continue
        seed = next(iter(degree))
        seen = {seed}
        stack = [seed]
        while stack:
            x = stack.pop()
            for y in graph[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        link_valid[vertex] = len(seen) == len(degree)
    return edges.astype(np.int32), incidence.astype(np.int32), incidence == 1, incidence > 2, orient, fc, link_valid


def face_geometry(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(length) & (length > 1e-12)
    normal = np.zeros_like(cross)
    normal[valid] = cross[valid] / length[valid, None]
    return 0.5 * length, normal, valid


def solid_angle_inside(points, vertices, faces, chunk=16):
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
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
        denominator = la * lb * lc + np.einsum("bij,bij->bi", a, b) * lc + np.einsum("bij,bij->bi", b, c) * la + np.einsum("bij,bij->bi", c, a) * lb
        angle = 2.0 * np.arctan2(numerator, denominator)
        winding = np.sum(angle, axis=1) / (4.0 * math.pi)
        result.append(np.abs(winding) > 0.5)
    return np.concatenate(result)


def point_triangle_distance_sq(points, tri):
    p = points[:, None, :]
    a, b, c = tri[None, :, 0, :], tri[None, :, 1, :], tri[None, :, 2, :]
    ab = b - a; ac = c - a; ap = p - a
    d1 = np.sum(ab * ap, -1); d2 = np.sum(ac * ap, -1)
    bp = p - b; d3 = np.sum(ab * bp, -1); d4 = np.sum(ac * bp, -1)
    cp = p - c; d5 = np.sum(ab * cp, -1); d6 = np.sum(ac * cp, -1)
    dist = np.full(d1.shape, np.inf)
    m = (d1 <= 0) & (d2 <= 0); dist[m] = np.sum(ap * ap, -1)[m]
    m = (d3 >= 0) & (d4 <= d3); dist[m] = np.minimum(dist[m], np.sum(bp * bp, -1)[m])
    vc = d1 * d4 - d3 * d2; m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    den = d1 - d3; t = np.divide(d1, den, out=np.zeros_like(d1), where=np.abs(den) > 1e-30); q = a + t[..., None] * ab; dist[m] = np.minimum(dist[m], np.sum((p - q) ** 2, -1)[m])
    m = (d6 >= 0) & (d5 <= d6); dist[m] = np.minimum(dist[m], np.sum(cp * cp, -1)[m])
    vb = d5 * d2 - d1 * d6; m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    den = d2 - d6; t = np.divide(d2, den, out=np.zeros_like(d2), where=np.abs(den) > 1e-30); q = a + t[..., None] * ac; dist[m] = np.minimum(dist[m], np.sum((p - q) ** 2, -1)[m])
    va = d3 * d6 - d5 * d4; m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    edge = c - b; den = (d4 - d3) + (d5 - d6); t = np.divide(d4 - d3, den, out=np.zeros_like(d4), where=np.abs(den) > 1e-30); q = b + t[..., None] * edge; dist[m] = np.minimum(dist[m], np.sum((p - q) ** 2, -1)[m])
    m = ~np.isfinite(dist); den = va + vb + vc; vv = np.divide(vb, den, out=np.zeros_like(vb), where=np.abs(den) > 1e-30); ww = np.divide(vc, den, out=np.zeros_like(vc), where=np.abs(den) > 1e-30); q = a + vv[..., None] * ab + ww[..., None] * ac; fallback = np.sum((p - q) ** 2, -1); dist[m] = fallback[m]
    return dist


def validate(manifest, maximum_surface_samples=128, maximum_distance_samples=48, maximum_sign_samples=96):
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("V32_GEOMETRY_V2_COURT_PARENT")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("V32_GEOMETRY_V2_COURT_SUBJECT")
    parents = dict(manifest.get("parents") or {})
    source_path = bound(parents.get("geometry_npz"), "V32_GEOMETRY_V2_COURT_SOURCE_DRIFT")
    snapshot_path = bound(parents.get("source_scene_snapshot"), "V32_GEOMETRY_V2_COURT_SNAPSHOT_DRIFT")
    outputs = dict(manifest.get("outputs") or {})
    topology_path = bound(outputs.get("topology"), "V32_GEOMETRY_V2_COURT_TOPOLOGY_DRIFT")
    surface_path = bound(outputs.get("surface"), "V32_GEOMETRY_V2_COURT_SURFACE_DRIFT")
    sign_path = bound(outputs.get("distance_sign"), "V32_GEOMETRY_V2_COURT_SIGN_DRIFT")
    with np.load(source_path, allow_pickle=False) as z:
        vsrc = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        owner = np.asarray(z["face_owner_privileged"], dtype=np.int32)
    snapshot = json.loads(snapshot_path.read_text())
    matrix = matrix4(snapshot["source_world_to_canonical"])
    vcan = transform(vsrc, matrix)
    raw_to_q, qsrc, qfaces, qowner = independent_quotient(vsrc, faces, owner)
    qcan = transform(qsrc, matrix)
    edges, incidence, boundary, nonmanifold, orient, fc, link = independent_topology(len(qsrc), qfaces)
    _, nsrc, valid_src = face_geometry(vsrc, faces)
    _, ncan, valid_can = face_geometry(vcan, faces)
    face_valid = valid_src & valid_can
    with np.load(topology_path, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "rest_vertices_canonical", "faces", "face_owner_privileged", "face_normal_source_world", "face_normal_canonical", "face_normal_valid", "raw_to_topology_vertex", "topology_vertices_source_world", "topology_vertices_canonical", "topology_faces", "topology_vertex_owner", "unique_edge_index", "edge_face_incidence_count", "boundary_edge", "nonmanifold_edge", "edge_orientation_consistent", "vertex_link_manifold", "face_component_id"}
        if not required.issubset(set(z.files)):
            raise ValueError("V32_GEOMETRY_V2_COURT_TOPOLOGY_ARRAYS")
        exact_arrays = {"rest_vertices_source_world": vsrc.astype(np.float32), "rest_vertices_canonical": vcan.astype(np.float32), "faces": faces.astype(np.int32), "face_owner_privileged": owner.astype(np.int32), "face_normal_source_world": nsrc.astype(np.float32), "face_normal_canonical": ncan.astype(np.float32), "face_normal_valid": face_valid.astype(np.bool_), "raw_to_topology_vertex": raw_to_q.astype(np.int32), "topology_vertices_source_world": qsrc.astype(np.float32), "topology_vertices_canonical": qcan.astype(np.float32), "topology_faces": qfaces.astype(np.int32), "topology_vertex_owner": qowner.astype(np.int32), "unique_edge_index": edges.astype(np.int32), "edge_face_incidence_count": incidence.astype(np.int32), "boundary_edge": boundary.astype(np.bool_), "nonmanifold_edge": nonmanifold.astype(np.bool_), "edge_orientation_consistent": orient.astype(np.bool_), "vertex_link_manifold": link.astype(np.bool_), "face_component_id": fc.astype(np.int32)}
        for name, expected in exact_arrays.items():
            if not np.array_equal(np.asarray(z[name]), expected):
                raise ValueError("V32_GEOMETRY_V2_COURT_TOPOLOGY_MISMATCH:" + name)
    if np.any(boundary) or np.any(nonmanifold) or not np.all(orient[incidence == 2]) or not np.all(link[np.unique(qfaces)]) or not np.all(face_valid):
        raise ValueError("V32_GEOMETRY_V2_COURT_SOURCE_NOT_CLOSED_MANIFOLD")
    with np.load(surface_path, allow_pickle=False) as z:
        tri_id = np.asarray(z["triangle_id"], dtype=np.int64); bary = np.asarray(z["barycentric"], dtype=np.float64); xyz = np.asarray(z["surface_xyz_canonical"], dtype=np.float64); normal = np.asarray(z["normal_canonical"], dtype=np.float64); normal_valid = np.asarray(z["normal_valid"], dtype=np.bool_)
    sample = np.unique(np.linspace(0, len(tri_id) - 1, min(maximum_surface_samples, len(tri_id)), dtype=np.int64))
    expected_xyz = np.sum(vcan[faces[tri_id[sample]]] * bary[sample, :, None], axis=1)
    if np.max(np.abs(expected_xyz - xyz[sample]), initial=0.0) > 5e-6 or np.max(np.abs(ncan[tri_id[sample]] - normal[sample]), initial=0.0) > 5e-6 or not np.all(normal_valid[sample]):
        raise ValueError("V32_GEOMETRY_V2_COURT_SURFACE_ADDRESS_OR_NORMAL")
    with np.load(sign_path, allow_pickle=False) as z:
        query = np.asarray(z["query_xyz_canonical"], dtype=np.float64); distance = np.asarray(z["unsigned_distance"], dtype=np.float64); closest_face = np.asarray(z["closest_face"], dtype=np.int64); closest_component = np.asarray(z["closest_topology_component"], dtype=np.int32); sign_valid = np.asarray(z["sign_valid"], dtype=np.bool_); occupancy = np.asarray(z["occupancy"], dtype=np.bool_); signed = np.asarray(z["signed_distance"], dtype=np.float64); union_valid = np.asarray(z["union_sign_valid"], dtype=np.bool_); union_occupancy = np.asarray(z["union_occupancy"], dtype=np.bool_); union_signed = np.asarray(z["union_signed_distance"], dtype=np.float64)
    if float(np.mean(sign_valid)) < 0.995 or float(np.mean(union_valid)) < 0.995:
        raise ValueError("V32_GEOMETRY_V2_COURT_SIGN_AUTHORITY_COVERAGE")
    distance_sample = np.unique(np.linspace(0, len(query) - 1, min(maximum_distance_samples, len(query)), dtype=np.int64))
    exact_sq = point_triangle_distance_sq(query[distance_sample], vcan[faces]); exact_distance = np.sqrt(np.min(exact_sq, axis=1))
    if np.max(np.abs(exact_distance - distance[distance_sample]), initial=0.0) > 2e-5:
        raise ValueError("V32_GEOMETRY_V2_COURT_DISTANCE_MISMATCH")
    valid_indices = np.flatnonzero(sign_valid & union_valid)
    sign_sample = valid_indices[np.unique(np.linspace(0, len(valid_indices) - 1, min(maximum_sign_samples, len(valid_indices)), dtype=np.int64))]
    component_expected = fc[closest_face[sign_sample]]
    if not np.array_equal(component_expected.astype(np.int32), closest_component[sign_sample]):
        raise ValueError("V32_GEOMETRY_V2_COURT_CLOSEST_COMPONENT")
    expected_nearest = np.zeros(len(sign_sample), dtype=np.bool_); expected_union = np.zeros(len(sign_sample), dtype=np.bool_)
    for component in np.unique(fc):
        fids = np.flatnonzero(fc == component); inside = solid_angle_inside(query[sign_sample], qcan, qfaces[fids]); expected_union |= inside; mask = component_expected == component; expected_nearest[mask] = inside[mask]
    if not np.array_equal(expected_nearest, occupancy[sign_sample]) or not np.array_equal(expected_union, union_occupancy[sign_sample]):
        raise ValueError("V32_GEOMETRY_V2_COURT_OCCUPANCY_MISMATCH")
    expected_signed = distance[sign_sample].copy(); expected_signed[expected_nearest] *= -1.0
    expected_union_signed = distance[sign_sample].copy(); expected_union_signed[expected_union] *= -1.0
    if np.max(np.abs(expected_signed - signed[sign_sample]), initial=0.0) > 1e-7 or np.max(np.abs(expected_union_signed - union_signed[sign_sample]), initial=0.0) > 1e-7:
        raise ValueError("V32_GEOMETRY_V2_COURT_SIGNED_DISTANCE_MISMATCH")
    payload = {"schema": SCHEMA, "status": PASS_STATUS, "subject_id": subject_id, "geometry_identity_sha256": str(manifest.get("geometry_identity_sha256") or ""), "topology": {"raw_vertex_count": int(len(vsrc)), "topology_vertex_count": int(len(qsrc)), "exact_duplicate_collapse_count": int(len(np.unique(faces)) - len(qsrc)), "component_count": int(len(np.unique(fc))), "boundary_edge_count": int(np.count_nonzero(boundary)), "nonmanifold_edge_count": int(np.count_nonzero(nonmanifold)), "vertex_link_manifold": True, "orientation_consistent": True}, "sign": {"nearest_component_valid_fraction": float(np.mean(sign_valid)), "union_valid_fraction": float(np.mean(union_valid)), "independent_method": "SOLID_ANGLE_GENERALIZED_WINDING_ABS_GT_HALF", "sampled_sign_rows": int(len(sign_sample))}, "surface": {"sampled_surface_rows": int(len(sample)), "sampled_distance_rows": int(len(distance_sample))}, "producer_functions_reused": False, "training_performed": False, "qualification_minted": False}
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--geometry-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    manifest = json.loads(args.geometry_manifest.resolve().read_text()); payload = validate(manifest); output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"); print(json.dumps({"status": payload["status"], "subject_id": payload["subject_id"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
