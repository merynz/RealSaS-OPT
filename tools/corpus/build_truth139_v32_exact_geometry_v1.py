#!/usr/bin/env python3
"""Build V3.2 exact-source geometry targets for one Truth139 subject.

Surface distance and surface addresses are exact with respect to the admitted rest
triangle mesh. Global sign/occupancy is fail-closed: it is emitted only when the
entire subject is closed, manifold and consistently oriented and independent ray
parity directions agree. F* is not read and is not an authority here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.Truth139V32ExactGeometry.v1"
STATUS = "PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def matrix4(value: object) -> np.ndarray:
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("V32_GEOMETRY_SOURCE_TO_CANONICAL_INVALID")
    if np.linalg.matrix_rank(m[:3, :3]) != 3:
        raise ValueError("V32_GEOMETRY_SOURCE_TO_CANONICAL_SINGULAR")
    return m


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    flat = p.reshape(-1, 3)
    h = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float64)], axis=1)
    out = (h @ m.T)[:, :3]
    return out.reshape(p.shape)


def transform_normals(normals: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    nm = np.linalg.inv(m[:3, :3]).T
    out = n @ nm.T
    length = np.linalg.norm(out, axis=1)
    valid = length > 1e-12
    result = np.zeros_like(out)
    result[valid] = out[valid] / length[valid, None]
    return result.reshape(np.asarray(normals).shape)


def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        v = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        f = np.asarray(z["faces"], dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("V32_GEOMETRY_VERTEX_LAYOUT")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("V32_GEOMETRY_FACE_LAYOUT")
    return v, f


def face_geometry(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dbl = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(dbl) & (dbl > 1e-12)
    n = np.zeros_like(cross)
    n[valid] = cross[valid] / dbl[valid, None]
    return 0.5 * dbl, n, valid


def topology(v_count: int, f: np.ndarray):
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    undirected = np.sort(directed, axis=1)
    edges, inverse, incidence = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    boundary = incidence == 1
    nonmanifold = incidence > 2

    orientation_consistent = np.ones(len(edges), dtype=np.bool_)
    for edge_id in np.flatnonzero(incidence == 2):
        ids = np.flatnonzero(inverse == edge_id)
        a, b = directed[ids[0]], directed[ids[1]]
        orientation_consistent[edge_id] = bool(a[0] == b[1] and a[1] == b[0])

    parent = np.arange(v_count, dtype=np.int64)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb: parent[rb] = ra
            else: parent[ra] = rb
    for a, b in edges.tolist():
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(v_count)], dtype=np.int64)
    lut = {r: i for i, r in enumerate(sorted(set(roots.tolist())))}
    vc = np.asarray([lut[int(r)] for r in roots], dtype=np.int32)
    fc = vc[f[:, 0]]
    if np.any(vc[f] != fc[:, None]):
        raise ValueError("V32_GEOMETRY_COMPONENT_INCONSISTENCY")
    return edges.astype(np.int32), incidence.astype(np.int32), boundary, nonmanifold, orientation_consistent, vc, fc.astype(np.int32)


def point_triangle_distance_sq(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Exact squared distance from P[N,3] to T[M,3,3], returned [N,M]."""
    p = points[:, None, :]
    a, b, c = tri[None, :, 0, :], tri[None, :, 1, :], tri[None, :, 2, :]
    ab, ac = b - a, c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1); d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1); d4 = np.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = np.sum(ab * cp, axis=-1); d6 = np.sum(ac * cp, axis=-1)

    dist = np.full(d1.shape, np.inf, dtype=np.float64)
    mask = (d1 <= 0) & (d2 <= 0); dist[mask] = np.sum(ap * ap, axis=-1)[mask]
    mask = (d3 >= 0) & (d4 <= d3); dist[mask] = np.minimum(dist[mask], np.sum(bp * bp, axis=-1)[mask])
    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    den = d1 - d3
    t = np.divide(d1, den, out=np.zeros_like(d1), where=np.abs(den) > 1e-30)
    q = a + t[..., None] * ab
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, axis=-1)[mask])
    mask = (d6 >= 0) & (d5 <= d6); dist[mask] = np.minimum(dist[mask], np.sum(cp * cp, axis=-1)[mask])
    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    den = d2 - d6
    t = np.divide(d2, den, out=np.zeros_like(d2), where=np.abs(den) > 1e-30)
    q = a + t[..., None] * ac
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, axis=-1)[mask])
    va = d3 * d6 - d5 * d4
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    e = c - b
    den = (d4 - d3) + (d5 - d6)
    t = np.divide(d4 - d3, den, out=np.zeros_like(d4), where=np.abs(den) > 1e-30)
    q = b + t[..., None] * e
    dist[mask] = np.minimum(dist[mask], np.sum((p - q) ** 2, axis=-1)[mask])

    face = ~(np.isfinite(dist))
    denom = va + vb + vc
    vbar = np.divide(vb, denom, out=np.zeros_like(vb), where=np.abs(denom) > 1e-30)
    wbar = np.divide(vc, denom, out=np.zeros_like(vc), where=np.abs(denom) > 1e-30)
    q = a + vbar[..., None] * ab + wbar[..., None] * ac
    face_dist = np.sum((p - q) ** 2, axis=-1)
    dist[face] = face_dist[face]
    return dist


def closest_exact(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, chunk: int = 64):
    tri = vertices[faces]
    out_d, out_face = [], []
    for start in range(0, len(points), chunk):
        d2 = point_triangle_distance_sq(points[start:start+chunk], tri)
        idx = np.argmin(d2, axis=1)
        out_face.append(idx.astype(np.int32))
        out_d.append(np.sqrt(d2[np.arange(len(idx)), idx]))
    return np.concatenate(out_d).astype(np.float32), np.concatenate(out_face).astype(np.int32)


def ray_parity(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, direction: np.ndarray, chunk: int = 64) -> np.ndarray:
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    d = np.asarray(direction, dtype=np.float64); d /= np.linalg.norm(d)
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.sum(e1 * h, axis=1)
    eps = 1e-10
    valid_face = np.abs(a) > eps
    inv_a = np.zeros_like(a); inv_a[valid_face] = 1.0 / a[valid_face]
    result = []
    for start in range(0, len(points), chunk):
        p = points[start:start+chunk]
        s = p[:, None, :] - tri[None, :, 0, :]
        u = np.sum(s * h[None, :, :], axis=-1) * inv_a[None, :]
        q = np.cross(s, e1[None, :, :])
        v = np.sum(q * d[None, None, :], axis=-1) * inv_a[None, :]
        t = np.sum(q * e2[None, :, :], axis=-1) * inv_a[None, :]
        hit = valid_face[None, :] & (u >= -1e-10) & (v >= -1e-10) & ((u + v) <= 1 + 1e-10) & (t > 1e-9)
        result.append((np.count_nonzero(hit, axis=1) % 2 == 1))
    return np.concatenate(result)


def sample_surface(subject_id: str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, valid: np.ndarray, count: int):
    ids = np.flatnonzero(valid)
    if not len(ids): raise ValueError("V32_GEOMETRY_ALL_FACES_DEGENERATE")
    area, _, _ = face_geometry(vertices, faces)
    prob = area[ids] / area[ids].sum()
    seed = int(hashlib.sha256((subject_id + "|surface").encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    face = rng.choice(ids, size=count, replace=True, p=prob).astype(np.int32)
    r1 = rng.random(count); r2 = rng.random(count); root = np.sqrt(r1)
    bary = np.stack([1-root, root*(1-r2), root*r2], axis=1)
    xyz = np.sum(vertices[faces[face]] * bary[..., None], axis=1)
    return face, bary.astype(np.float32), xyz.astype(np.float32), normals[face].astype(np.float32)


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def build(subject_id: str, geometry_npz: Path, snapshot_json: Path, camera_json: Path, output_dir: Path,
          surface_samples: int = 32768, query_samples: int = 16384) -> dict:
    geometry_npz = geometry_npz.resolve(); snapshot_json = snapshot_json.resolve(); camera_json = camera_json.resolve(); output_dir = output_dir.resolve()
    snapshot = json.loads(snapshot_json.read_text()); camera = json.loads(camera_json.read_text())
    vertices_sw, faces = load_geometry(geometry_npz)
    m = matrix4(snapshot["source_world_to_canonical"])
    vertices = transform_points(vertices_sw, m)
    area, normals_sw, face_valid = face_geometry(vertices_sw, faces)
    normals = transform_normals(normals_sw, m)
    edges, incidence, boundary, nonmanifold, orient_ok, vc, fc = topology(len(vertices), faces)
    orientation_valid = bool(np.all(orient_ok[incidence == 2]))
    global_sign_topology_valid = bool(not np.any(boundary) and not np.any(nonmanifold) and orientation_valid and np.all(face_valid))

    sface, sbary, sxyz, snormal = sample_surface(subject_id, vertices, faces, normals, face_valid, int(surface_samples))
    seed = int(hashlib.sha256((subject_id + "|query").encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    near_n = query_samples // 2
    ids = rng.integers(0, len(sxyz), size=near_n)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    offsets = rng.uniform(-0.03, 0.03, size=near_n) * max(diag, 1e-6)
    near = sxyz[ids].astype(np.float64) + snormal[ids].astype(np.float64) * offsets[:, None]
    lo = np.minimum(vertices.min(axis=0), np.asarray([-1, -1, -1], dtype=np.float64))
    hi = np.maximum(vertices.max(axis=0), np.asarray([1, 1, 1], dtype=np.float64))
    uniform = rng.uniform(lo, hi, size=(query_samples - near_n, 3))
    query = np.concatenate([near, uniform], axis=0).astype(np.float64)
    distance, closest_face = closest_exact(query, vertices, faces)

    directions = np.asarray([[1.0, 0.173, 0.319], [0.271, 1.0, 0.113], [0.197, 0.233, 1.0]], dtype=np.float64)
    parity = np.stack([ray_parity(query, vertices, faces, d) for d in directions], axis=1)
    per_query_agree = np.all(parity == parity[:, :1], axis=1)
    sign_valid = np.full(len(query), global_sign_topology_valid, dtype=np.bool_) & per_query_agree
    occupancy = parity[:, 0].astype(np.bool_)
    signed_distance = distance.copy(); signed_distance[occupancy] *= -1.0
    signed_distance[~sign_valid] = 0.0

    normal_valid = face_valid & orientation_valid
    topology_path = output_dir / "V32_EXACT_GEOMETRY_TOPOLOGY.npz"
    write_npz(topology_path,
        rest_vertices_source_world=vertices_sw.astype(np.float32), rest_vertices_canonical=vertices.astype(np.float32), faces=faces.astype(np.int32),
        face_normal_source_world=normals_sw.astype(np.float32), face_normal_canonical=normals.astype(np.float32), face_normal_valid=normal_valid.astype(np.bool_),
        face_area_source_world=area.astype(np.float32), nondegenerate_face=face_valid.astype(np.bool_), unique_edge_index=edges,
        edge_face_incidence_count=incidence, boundary_edge=boundary.astype(np.bool_), nonmanifold_edge=nonmanifold.astype(np.bool_),
        edge_orientation_consistent=orient_ok.astype(np.bool_), vertex_component_id=vc, face_component_id=fc)
    surface_path = output_dir / "V32_EXACT_SURFACE_SAMPLES.npz"
    write_npz(surface_path, triangle_id=sface, barycentric=sbary, surface_xyz_canonical=sxyz, normal_canonical=snormal,
              normal_valid=normal_valid[sface].astype(np.bool_))
    query_path = output_dir / "V32_EXACT_DISTANCE_SIGN_TARGETS.npz"
    write_npz(query_path, query_xyz_canonical=query.astype(np.float32), unsigned_distance=distance.astype(np.float32),
              closest_face=closest_face, sign_valid=sign_valid, occupancy=occupancy, signed_distance=signed_distance.astype(np.float32))

    payload = {
        "schema": SCHEMA, "status": STATUS, "subject_id": subject_id,
        "parents": {
            "geometry_npz": {"path": str(geometry_npz), "sha256": sha256(geometry_npz)},
            "source_scene_snapshot": {"path": str(snapshot_json), "sha256": sha256(snapshot_json)},
            "camera_bundle": {"path": str(camera_json), "sha256": sha256(camera_json)},
            "source_sha256": snapshot.get("source_sha256")
        },
        "outputs": {
            "topology": {"path": str(topology_path), "sha256": sha256(topology_path)},
            "surface": {"path": str(surface_path), "sha256": sha256(surface_path)},
            "distance_sign": {"path": str(query_path), "sha256": sha256(query_path)}
        },
        "counts": {
            "vertex_count": int(len(vertices)), "face_count": int(len(faces)), "component_count": int(len(np.unique(vc))),
            "boundary_edge_count": int(np.count_nonzero(boundary)), "nonmanifold_edge_count": int(np.count_nonzero(nonmanifold)),
            "surface_sample_count": int(surface_samples), "query_sample_count": int(query_samples),
            "sign_valid_query_count": int(np.count_nonzero(sign_valid))
        },
        "validity": {
            "orientation_consistent": orientation_valid,
            "global_sign_topology_valid": global_sign_topology_valid,
            "global_sign_policy": "FAIL_CLOSED_ALL_COMPONENTS_REQUIRED",
            "parity_directions": directions.tolist()
        },
        "authority": {
            "fstar_used": false if False else False,
            "surface_distance_authority": "EXACT_SOURCE_TRIANGLE_MESH",
            "global_sign_fabricated_for_invalid_mesh": False,
            "training_performed": False,
            "qualification_minted": False
        }
    }
    payload["geometry_identity_sha256"] = digest(payload)
    manifest = output_dir / "V32_EXACT_GEOMETRY_MANIFEST.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": STATUS, "subject_id": subject_id, "sign_valid_query_count": payload["counts"]["sign_valid_query_count"], "geometry_identity_sha256": payload["geometry_identity_sha256"]}, sort_keys=True), flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-id", required=True); p.add_argument("--geometry-npz", type=Path, required=True)
    p.add_argument("--source-scene-snapshot", type=Path, required=True); p.add_argument("--camera-bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--surface-samples", type=int, default=32768); p.add_argument("--query-samples", type=int, default=16384)
    a = p.parse_args(); build(a.subject_id, a.geometry_npz, a.source_scene_snapshot, a.camera_bundle, a.output_dir, a.surface_samples, a.query_samples)

if __name__ == "__main__": main()
