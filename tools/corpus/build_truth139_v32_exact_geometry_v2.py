#!/usr/bin/env python3
"""Build Truth139 V3.2 geometry truth with geometric/topological vertex identity.

V2 fixes a scientific bug in V1: raw render/import vertex indices were treated as
geometric topology. glTF/DCC transports legitimately split one geometric vertex
at UV/material/normal seams. V2 preserves those raw vertices for surface
addressing but builds topology/sign authority on an owner-aware exact-position
quotient.

Both nearest-component and union occupancy are materialized. The canonical
signed-distance target is nearest-component signed distance because it preserves
zero-crossing evidence for every closed authored surface, including nested
components. Union occupancy is retained separately for solid-envelope courts.
"""
from __future__ import annotations

import argparse, hashlib, json
from dataclasses import replace
from pathlib import Path
import numpy as np

from models.iris.truth.geometric_topology_v1 import build_geometric_topology_v1

SCHEMA = "RealSaS.Truth139V32ExactGeometry.v2"
STATUS = "PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2__INDEPENDENT_COURT_REQUIRED"


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


def matrix4(value):
    m = np.asarray(value, dtype=np.float64)
    if (
        m.shape != (4, 4)
        or not np.isfinite(m).all()
        or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or np.linalg.matrix_rank(m[:3, :3]) != 3
    ):
        raise ValueError("V32_GEOMETRY_V2_SOURCE_TO_CANONICAL_INVALID")
    return m


def transform_points(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    flat = p.reshape(-1, 3)
    h = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    return (h @ matrix.T)[:, :3].reshape(p.shape)


def load_geometry(path):
    with np.load(path, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "faces", "face_owner_privileged"}
        if not required.issubset(set(z.files)):
            raise ValueError("V32_GEOMETRY_V2_SOURCE_ARRAYS_MISSING")
        v = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        f = np.asarray(z["faces"], dtype=np.int64)
        owner = np.asarray(z["face_owner_privileged"], dtype=np.int32)
    if (
        v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
        or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
        or np.any(f < 0) or np.any(f >= len(v))
        or owner.shape != (len(f),) or np.any(owner < 0)
    ):
        raise ValueError("V32_GEOMETRY_V2_SOURCE_LAYOUT")
    return v, f, owner


def face_geometry(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-12)
    normal = np.zeros_like(cross)
    normal[valid] = cross[valid] / double_area[valid, None]
    return 0.5 * double_area, normal, valid


def ray_parity(points, vertices, faces, direction, chunk=32):
    """Direct triangle parity for one closed component."""
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    a = np.sum(e1 * h, axis=1)
    valid_face = np.abs(a) > 1e-10
    inv = np.zeros_like(a)
    inv[valid_face] = 1.0 / a[valid_face]
    result = []
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        s = p[:, None, :] - tri[None, :, 0, :]
        u = np.sum(s * h[None, :, :], axis=-1) * inv[None, :]
        q = np.cross(s, e1[None, :, :])
        vv = np.sum(q * direction[None, None, :], axis=-1) * inv[None, :]
        t = np.sum(q * e2[None, :, :], axis=-1) * inv[None, :]
        hit = (
            valid_face[None, :]
            & (u >= -1e-10)
            & (vv >= -1e-10)
            & ((u + vv) <= 1.0 + 1e-10)
            & (t > 1e-9)
        )
        result.append(np.count_nonzero(hit, axis=1) % 2 == 1)
    return np.concatenate(result)


def component_parity_fields(points, topology, closest_face):
    """Three-direction sign court, separately per topological component."""
    directions = (
        np.asarray([1.0, 0.173, 0.319]),
        np.asarray([0.271, 1.0, 0.113]),
        np.asarray([0.431, 0.719, 1.0]),
    )
    components = np.unique(topology.face_component_id)
    closest_component = topology.face_component_id[np.asarray(closest_face, dtype=np.int64)]
    union_votes = np.zeros((len(points), len(directions)), dtype=np.bool_)
    nearest_votes = np.zeros((len(points), len(directions)), dtype=np.bool_)

    for component in components:
        face_ids = np.flatnonzero(topology.face_component_id == component)
        component_faces = topology.topology_faces[face_ids]
        votes = np.stack(
            [
                ray_parity(points, topology.topology_vertices, component_faces, direction)
                for direction in directions
            ],
            axis=1,
        )
        union_votes |= votes
        mask = closest_component == component
        nearest_votes[mask] = votes[mask]

    nearest_valid = np.all(nearest_votes == nearest_votes[:, :1], axis=1)
    union_valid = np.all(union_votes == union_votes[:, :1], axis=1)
    return (
        closest_component.astype(np.int32),
        nearest_valid.astype(np.bool_),
        nearest_votes[:, 0].astype(np.bool_),
        union_valid.astype(np.bool_),
        union_votes[:, 0].astype(np.bool_),
    )


def sample_surface(subject, vertices, faces, normals, valid, count):
    area, _, _ = face_geometry(vertices, faces)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError("V32_GEOMETRY_V2_NO_VALID_SURFACE")
    rng = np.random.default_rng(
        int(hashlib.sha256((subject + "|surface-v2").encode()).hexdigest()[:16], 16)
    )
    probability = area[ids] / area[ids].sum()
    face = rng.choice(ids, size=count, replace=True, p=probability).astype(np.int32)
    r1 = rng.random(count)
    r2 = rng.random(count)
    root = np.sqrt(r1)
    barycentric = np.stack((1 - root, root * (1 - r2), root * r2), axis=1)
    xyz = np.sum(vertices[faces[face]] * barycentric[..., None], axis=1)
    return face, barycentric.astype(np.float32), xyz.astype(np.float32), normals[face].astype(np.float32)


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def build(subject_id, geometry_npz, snapshot_json, camera_json, output_dir, surface_samples=32768, query_samples=8192):
    geometry_npz = geometry_npz.resolve()
    snapshot_json = snapshot_json.resolve()
    camera_json = camera_json.resolve()
    output_dir = output_dir.resolve()
    snapshot = json.loads(snapshot_json.read_text())

    vertices_source, faces, face_owner = load_geometry(geometry_npz)
    matrix = matrix4(snapshot["source_world_to_canonical"])
    vertices_canonical = transform_points(vertices_source, matrix)

    area_source, normal_source, face_valid_source = face_geometry(vertices_source, faces)
    _, normal_canonical, face_valid_canonical = face_geometry(vertices_canonical, faces)
    face_valid = face_valid_source & face_valid_canonical

    topology_source = build_geometric_topology_v1(vertices_source, faces, face_owner)
    topology_vertices_canonical = transform_points(topology_source.topology_vertices, matrix)
    topology_canonical = build_geometric_topology_v1(vertices_canonical, faces, face_owner)
    if not np.array_equal(topology_source.topology_faces, topology_canonical.topology_faces):
        raise ValueError("V32_GEOMETRY_V2_TOPOLOGY_REMAP_TRANSFORM_DRIFT")

    orientation_valid = bool(
        np.all(topology_source.edge_orientation_consistent[topology_source.edge_face_incidence_count == 2])
    )
    topology_sign_valid = bool(
        topology_source.sign_topology_valid and orientation_valid and np.all(face_valid)
    )

    surface_face, surface_bary, surface_xyz, surface_normal = sample_surface(
        subject_id, vertices_canonical, faces, normal_canonical, face_valid, int(surface_samples)
    )

    rng = np.random.default_rng(
        int(hashlib.sha256((subject_id + "|query-v2").encode()).hexdigest()[:16], 16)
    )
    near_count = query_samples // 2
    sampled_surface = rng.integers(0, len(surface_xyz), size=near_count)
    diagonal = float(np.linalg.norm(vertices_canonical.max(0) - vertices_canonical.min(0)))
    offset = rng.uniform(-0.03, 0.03, size=near_count) * max(diagonal, 1e-6)
    near = surface_xyz[sampled_surface].astype(np.float64) + surface_normal[sampled_surface].astype(np.float64) * offset[:, None]
    lo = np.minimum(vertices_canonical.min(0), [-1, -1, -1])
    hi = np.maximum(vertices_canonical.max(0), [1, 1, 1])
    query = np.concatenate((near, rng.uniform(lo, hi, size=(query_samples - near_count, 3)))).astype(np.float64)

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("V32_GEOMETRY_V2_TRIMESH_REQUIRED") from exc
    raw_mesh = trimesh.Trimesh(vertices=vertices_canonical, faces=faces, process=False)
    try:
        _, distance, closest_face = trimesh.proximity.closest_point(raw_mesh, query)
    except ModuleNotFoundError:
        _, distance, closest_face = trimesh.proximity.closest_point_naive(raw_mesh, query)
    distance = np.asarray(distance, dtype=np.float32)
    closest_face = np.asarray(closest_face, dtype=np.int32)

    closest_component = np.full(len(query), -1, dtype=np.int32)
    nearest_valid = np.zeros(len(query), dtype=np.bool_)
    nearest_inside = np.zeros(len(query), dtype=np.bool_)
    union_valid = np.zeros(len(query), dtype=np.bool_)
    union_inside = np.zeros(len(query), dtype=np.bool_)
    if topology_sign_valid:
        canonical_topology = replace(topology_source, topology_vertices=topology_vertices_canonical)
        closest_component, nearest_valid, nearest_inside, union_valid, union_inside = component_parity_fields(
            query, canonical_topology, closest_face
        )

    signed_distance = np.zeros(len(query), dtype=np.float32)
    signed_distance[nearest_valid] = distance[nearest_valid]
    signed_distance[nearest_valid & nearest_inside] *= -1.0
    union_signed_distance = np.zeros(len(query), dtype=np.float32)
    union_signed_distance[union_valid] = distance[union_valid]
    union_signed_distance[union_valid & union_inside] *= -1.0

    raw_vertex_component = np.full(len(vertices_source), -1, dtype=np.int32)
    mapped = topology_source.raw_to_topology_vertex >= 0
    raw_vertex_component[mapped] = topology_source.topology_vertex_component_id[topology_source.raw_to_topology_vertex[mapped]]

    topology_path = output_dir / "V32_EXACT_GEOMETRY_TOPOLOGY_V2.npz"
    write_npz(
        topology_path,
        rest_vertices_source_world=vertices_source.astype(np.float32),
        rest_vertices_canonical=vertices_canonical.astype(np.float32),
        faces=faces.astype(np.int32),
        face_owner_privileged=face_owner.astype(np.int32),
        face_normal_source_world=normal_source.astype(np.float32),
        face_normal_canonical=normal_canonical.astype(np.float32),
        face_normal_valid=face_valid.astype(np.bool_),
        face_area_source_world=area_source.astype(np.float32),
        nondegenerate_face=face_valid.astype(np.bool_),
        raw_to_topology_vertex=topology_source.raw_to_topology_vertex.astype(np.int32),
        topology_vertices_source_world=topology_source.topology_vertices.astype(np.float32),
        topology_vertices_canonical=topology_vertices_canonical.astype(np.float32),
        topology_faces=topology_source.topology_faces.astype(np.int32),
        topology_vertex_owner=topology_source.topology_vertex_owner.astype(np.int32),
        unique_edge_index=topology_source.unique_edge_index.astype(np.int32),
        edge_face_incidence_count=topology_source.edge_face_incidence_count.astype(np.int32),
        boundary_edge=topology_source.boundary_edge.astype(np.bool_),
        nonmanifold_edge=topology_source.nonmanifold_edge.astype(np.bool_),
        edge_orientation_consistent=topology_source.edge_orientation_consistent.astype(np.bool_),
        vertex_link_manifold=topology_source.vertex_link_manifold.astype(np.bool_),
        vertex_component_id=raw_vertex_component,
        topology_vertex_component_id=topology_source.topology_vertex_component_id.astype(np.int32),
        face_component_id=topology_source.face_component_id.astype(np.int32),
    )

    surface_path = output_dir / "V32_EXACT_SURFACE_SAMPLES_V2.npz"
    write_npz(
        surface_path,
        triangle_id=surface_face,
        barycentric=surface_bary,
        surface_xyz_canonical=surface_xyz,
        normal_canonical=surface_normal,
        normal_valid=face_valid[surface_face].astype(np.bool_),
    )

    query_path = output_dir / "V32_EXACT_DISTANCE_SIGN_TARGETS_V2.npz"
    write_npz(
        query_path,
        query_xyz_canonical=query.astype(np.float32),
        unsigned_distance=distance,
        closest_face=closest_face,
        closest_topology_component=closest_component,
        sign_valid=nearest_valid,
        occupancy=nearest_inside,
        signed_distance=signed_distance,
        union_sign_valid=union_valid,
        union_occupancy=union_inside,
        union_signed_distance=union_signed_distance,
    )

    counts = {
        "raw_vertex_count": int(len(vertices_source)),
        "referenced_raw_vertex_count": int(len(np.unique(faces))),
        "topology_vertex_count": int(len(topology_source.topology_vertices)),
        "exact_duplicate_collapse_count": int(topology_source.exact_duplicate_collapse_count),
        "face_count": int(len(faces)),
        "component_count": int(len(np.unique(topology_source.face_component_id))),
        "boundary_edge_count": int(np.count_nonzero(topology_source.boundary_edge)),
        "nonmanifold_edge_count": int(np.count_nonzero(topology_source.nonmanifold_edge)),
        "nonmanifold_vertex_count": int(np.count_nonzero(~topology_source.vertex_link_manifold[np.unique(topology_source.topology_faces)])),
        "surface_sample_count": int(surface_samples),
        "query_sample_count": int(query_samples),
        "sign_valid_query_count": int(np.count_nonzero(nearest_valid)),
        "union_sign_valid_query_count": int(np.count_nonzero(union_valid)),
    }
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "parents": {
            "geometry_npz": {"path": str(geometry_npz), "sha256": sha256(geometry_npz)},
            "source_scene_snapshot": {"path": str(snapshot_json), "sha256": sha256(snapshot_json)},
            "camera_bundle": {"path": str(camera_json), "sha256": sha256(camera_json)},
            "source_sha256": snapshot.get("source_sha256"),
        },
        "outputs": {
            "topology": {"path": str(topology_path), "sha256": sha256(topology_path)},
            "surface": {"path": str(surface_path), "sha256": sha256(surface_path)},
            "distance_sign": {"path": str(query_path), "sha256": sha256(query_path)},
        },
        "counts": counts,
        "validity": {
            "orientation_consistent": orientation_valid,
            "closed_edge_manifold": topology_source.closed_edge_manifold,
            "closed_vertex_manifold": topology_source.closed_vertex_manifold,
            "global_sign_topology_valid": topology_sign_valid,
            "nearest_component_three_ray_agreement_fraction": float(np.mean(nearest_valid)),
            "union_three_ray_agreement_fraction": float(np.mean(union_valid)),
        },
        "authority": {
            "fstar_used": False,
            "surface_distance_authority": "EXACT_SOURCE_TRIANGLE_MESH",
            "topology_identity": "OWNER_PLUS_EXACT_SOURCE_POSITION_QUOTIENT",
            "render_vertex_identity_used_as_topology": False,
            "tolerance_weld_used": False,
            "signed_distance_authority": "NEAREST_CLOSED_TOPOLOGICAL_COMPONENT_THREE_RAY_CONSENSUS",
            "union_occupancy_materialized": True,
            "global_sign_fabricated_for_invalid_mesh": False,
            "training_performed": False,
            "qualification_minted": False,
        },
        "topology_identity_sha256": topology_source.identity_sha256,
    }
    payload["geometry_identity_sha256"] = digest(payload)
    manifest = output_dir / "V32_EXACT_GEOMETRY_MANIFEST_V2.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": STATUS,
        "subject_id": subject_id,
        "topology_vertex_count": counts["topology_vertex_count"],
        "component_count": counts["component_count"],
        "boundary_edge_count": counts["boundary_edge_count"],
        "sign_valid_query_count": counts["sign_valid_query_count"],
        "geometry_identity_sha256": payload["geometry_identity_sha256"],
    }, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--source-scene-snapshot", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=32768)
    parser.add_argument("--query-samples", type=int, default=8192)
    args = parser.parse_args()
    build(args.subject_id, args.geometry_npz, args.source_scene_snapshot, args.camera_bundle, args.output_dir, args.surface_samples, args.query_samples)


if __name__ == "__main__":
    main()
