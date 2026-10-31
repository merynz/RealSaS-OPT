#!/usr/bin/env python3
from __future__ import annotations

"""Build component-scoped surface/sign truth from immutable source geometry.

Unlike the historical Truth139 V2 producer, this producer never requires every
surface component in a subject to be a closed manifold. Unsigned distance remains
available for qualified nondegenerate source surfaces. Signed distance/occupancy is
stored only for query indices whose nearest sign component is independently closed
under the producer topology and whose three parity probes agree.

There is intentionally no dense placeholder signed target for invalid queries.
Unknown sign is represented by absence from signed_query_index. An independent
solid-angle court must pass before any family authority can be minted.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.component_geometry_authority_v1 import (
    build_component_geometry_authority_v1,
)


SCHEMA = "RealSaS.ComponentGeometryAuthority.v1"
STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


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
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or np.linalg.matrix_rank(matrix[:3, :3]) != 3
    ):
        raise ValueError("COMPONENT_GEOMETRY_SOURCE_TO_CANONICAL_INVALID")
    return matrix


def transform(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    return (homogeneous @ matrix.T)[:, :3].reshape(points.shape)


def load_source(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "faces", "face_owner_privileged"}
        if not required.issubset(z.files):
            raise ValueError("COMPONENT_GEOMETRY_SOURCE_ARRAYS_MISSING")
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        owner = np.asarray(z["face_owner_privileged"], dtype=np.int32)
    return vertices, faces, owner


def face_geometry(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(length) & (length > 1.0e-12)
    normal = np.zeros_like(cross)
    normal[valid] = cross[valid] / length[valid, None]
    return 0.5 * length, normal, valid


def ray_parity(points, vertices, faces, direction, chunk=32):
    tri = vertices[faces]
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    a = np.sum(e1 * h, axis=1)
    face_ok = np.abs(a) > 1.0e-10
    inv = np.zeros_like(a)
    inv[face_ok] = 1.0 / a[face_ok]
    result = []
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        s = p[:, None, :] - tri[None, :, 0, :]
        u = np.sum(s * h[None, :, :], axis=-1) * inv[None, :]
        q = np.cross(s, e1[None, :, :])
        vv = np.sum(q * direction[None, None, :], axis=-1) * inv[None, :]
        t = np.sum(q * e2[None, :, :], axis=-1) * inv[None, :]
        hit = (
            face_ok[None, :]
            & (u >= -1.0e-10)
            & (vv >= -1.0e-10)
            & ((u + vv) <= 1.0 + 1.0e-10)
            & (t > 1.0e-9)
        )
        result.append((np.count_nonzero(hit, axis=1) % 2) == 1)
    return np.concatenate(result) if result else np.zeros(0, dtype=np.bool_)


def sample_surface(subject_id, vertices, faces, face_valid, normal, count):
    area, _, _ = face_geometry(vertices, faces)
    ids = np.flatnonzero(face_valid)
    if not len(ids):
        raise ValueError("COMPONENT_GEOMETRY_NO_VALID_SURFACE")
    probability = area[ids] / area[ids].sum()
    rng = np.random.default_rng(
        int(hashlib.sha256((subject_id + "|surface-authority-v1").encode()).hexdigest()[:16], 16)
    )
    selected = rng.choice(ids, size=int(count), replace=True, p=probability).astype(np.int32)
    r1 = rng.random(int(count))
    r2 = rng.random(int(count))
    root = np.sqrt(r1)
    bary = np.stack((1.0 - root, root * (1.0 - r2), root * r2), axis=1)
    xyz = np.sum(vertices[faces[selected]] * bary[..., None], axis=1)
    return selected, bary.astype(np.float32), xyz.astype(np.float32), normal[selected].astype(np.float32)


def query_points(subject_id, vertices, surface_xyz, surface_normal, count):
    rng = np.random.default_rng(
        int(hashlib.sha256((subject_id + "|query-authority-v1").encode()).hexdigest()[:16], 16)
    )
    count = int(count)
    near_count = count // 2
    chosen = rng.integers(0, len(surface_xyz), size=near_count)
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    offsets = rng.uniform(-0.03, 0.03, size=near_count) * max(diagonal, 1.0e-6)
    near = surface_xyz[chosen].astype(np.float64) + surface_normal[chosen].astype(np.float64) * offsets[:, None]
    lo = np.minimum(vertices.min(0), [-1.0, -1.0, -1.0])
    hi = np.maximum(vertices.max(0), [1.0, 1.0, 1.0])
    far = rng.uniform(lo, hi, size=(count - near_count, 3))
    return np.concatenate((near, far), axis=0).astype(np.float64)


def closest_valid_surface(query, vertices, faces, face_valid):
    valid_face_ids = np.flatnonzero(face_valid)
    if not len(valid_face_ids):
        raise ValueError("COMPONENT_GEOMETRY_NO_DISTANCE_SURFACE")
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("COMPONENT_GEOMETRY_TRIMESH_REQUIRED") from exc
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[valid_face_ids], process=False)
    try:
        _, distance, local_face = trimesh.proximity.closest_point(mesh, query)
    except ModuleNotFoundError:
        _, distance, local_face = trimesh.proximity.closest_point_naive(mesh, query)
    closest = valid_face_ids[np.asarray(local_face, dtype=np.int64)]
    distance = np.asarray(distance, dtype=np.float64)
    if not np.isfinite(distance).all():
        raise ValueError("COMPONENT_GEOMETRY_NONFINITE_DISTANCE")
    return distance.astype(np.float32), closest.astype(np.int32)


def compact_sign_targets(query, distance, closest_face, authority, canonical_topology_vertices):
    closest_component = authority.sign_component_id[closest_face]
    component_valid_for_query = np.zeros(len(query), dtype=np.bool_)
    mapped = closest_component >= 0
    component_valid_for_query[mapped] = authority.sign_component_valid[closest_component[mapped]]

    signed_indices = []
    occupancy_values = []
    signed_values = []
    signed_components = []
    directions = (
        np.asarray([1.0, 0.173, 0.319]),
        np.asarray([0.271, 1.0, 0.113]),
        np.asarray([0.431, 0.719, 1.0]),
    )
    for component_id in np.flatnonzero(authority.sign_component_valid):
        indices = np.flatnonzero(component_valid_for_query & (closest_component == component_id))
        if not len(indices):
            continue
        face_ids = np.flatnonzero(authority.sign_component_id == component_id)
        component_faces = authority.topology_faces[face_ids]
        votes = np.stack(
            [ray_parity(query[indices], canonical_topology_vertices, component_faces, d) for d in directions],
            axis=1,
        )
        consensus = np.all(votes == votes[:, :1], axis=1)
        kept = indices[consensus]
        inside = votes[consensus, 0]
        values = distance[kept].astype(np.float32).copy()
        values[inside] *= -1.0
        signed_indices.append(kept.astype(np.int32))
        occupancy_values.append(inside.astype(np.bool_))
        signed_values.append(values)
        signed_components.append(np.full(len(kept), component_id, dtype=np.int32))

    if signed_indices:
        index = np.concatenate(signed_indices)
        order = np.argsort(index)
        return (
            index[order],
            np.concatenate(signed_values)[order],
            np.concatenate(occupancy_values)[order],
            np.concatenate(signed_components)[order],
            closest_component.astype(np.int32),
            component_valid_for_query.astype(np.bool_),
        )
    return (
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.bool_),
        np.zeros(0, dtype=np.int32),
        closest_component.astype(np.int32),
        component_valid_for_query.astype(np.bool_),
    )


def write_npz(path: Path, **arrays):
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype == object or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()):
            raise ValueError("COMPONENT_GEOMETRY_INVALID_OUTPUT_ARRAY:" + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(subject_id, geometry_npz, snapshot_json, camera_json, output_dir, *, surface_samples=32768, query_samples=8192):
    geometry_npz = Path(geometry_npz).resolve()
    snapshot_json = Path(snapshot_json).resolve()
    camera_json = Path(camera_json).resolve()
    output_dir = Path(output_dir).resolve()
    snapshot = json.loads(snapshot_json.read_text())
    json.loads(camera_json.read_text())  # byte/hash-bound input even though sampling is source-space/canonical.

    vertices_source, faces, owner = load_source(geometry_npz)
    matrix = matrix4(snapshot["source_world_to_canonical"])
    vertices_canonical = transform(vertices_source, matrix)
    authority = build_component_geometry_authority_v1(vertices_source, faces, owner)
    topology_vertices_canonical = transform(authority.topology_vertices, matrix)

    _, normal_source, source_normal_valid = face_geometry(vertices_source, faces)
    _, normal_canonical, canonical_normal_valid = face_geometry(vertices_canonical, faces)
    face_valid = authority.face_valid & source_normal_valid & canonical_normal_valid
    if not np.array_equal(face_valid, authority.face_valid):
        raise ValueError("COMPONENT_GEOMETRY_CANONICAL_FACE_VALIDITY_DRIFT")

    surface_face, surface_bary, surface_xyz, surface_normal = sample_surface(
        subject_id, vertices_canonical, faces, face_valid, normal_canonical, surface_samples
    )
    query = query_points(subject_id, vertices_canonical, surface_xyz, surface_normal, query_samples)
    unsigned_distance, closest_face = closest_valid_surface(query, vertices_canonical, faces, face_valid)
    (
        signed_query_index,
        signed_distance_value,
        occupancy_value,
        signed_component_id,
        query_nearest_sign_component,
        query_nearest_component_is_sign_candidate,
    ) = compact_sign_targets(
        query,
        unsigned_distance,
        closest_face,
        authority,
        topology_vertices_canonical,
    )

    topology_path = output_dir / "COMPONENT_GEOMETRY_TOPOLOGY_V1.npz"
    write_npz(
        topology_path,
        rest_vertices_source_world=vertices_source.astype(np.float32),
        rest_vertices_canonical=vertices_canonical.astype(np.float32),
        faces=faces.astype(np.int32),
        face_owner_privileged=owner.astype(np.int32),
        face_normal_source_world=normal_source.astype(np.float32),
        face_normal_canonical=normal_canonical.astype(np.float32),
        face_valid=face_valid.astype(np.bool_),
        raw_to_topology_vertex=authority.raw_to_topology_vertex.astype(np.int32),
        topology_vertices_source_world=authority.topology_vertices.astype(np.float32),
        topology_vertices_canonical=topology_vertices_canonical.astype(np.float32),
        topology_faces=authority.topology_faces.astype(np.int32),
        topology_vertex_owner=authority.topology_vertex_owner.astype(np.int32),
        surface_component_id=authority.surface_component_id.astype(np.int32),
        sign_component_id=authority.sign_component_id.astype(np.int32),
        sign_component_valid=authority.sign_component_valid.astype(np.bool_),
    )

    surface_path = output_dir / "COMPONENT_SURFACE_TARGETS_V1.npz"
    write_npz(
        surface_path,
        triangle_id=surface_face,
        barycentric=surface_bary,
        surface_xyz_canonical=surface_xyz,
        normal_canonical=surface_normal,
        normal_valid=face_valid[surface_face].astype(np.bool_),
    )

    query_path = output_dir / "COMPONENT_DISTANCE_SIGN_TARGETS_V1.npz"
    write_npz(
        query_path,
        query_xyz_canonical=query.astype(np.float32),
        unsigned_distance=unsigned_distance.astype(np.float32),
        unsigned_distance_valid=np.ones(len(query), dtype=np.bool_),
        closest_face=closest_face.astype(np.int32),
        query_nearest_sign_component=query_nearest_sign_component.astype(np.int32),
        query_nearest_component_is_sign_candidate=query_nearest_component_is_sign_candidate.astype(np.bool_),
        signed_query_index=signed_query_index.astype(np.int32),
        signed_distance_value=signed_distance_value.astype(np.float32),
        occupancy_value=occupancy_value.astype(np.bool_),
        signed_component_id=signed_component_id.astype(np.int32),
    )

    valid_component_count = int(np.count_nonzero(authority.sign_component_valid))
    counts = {
        "raw_vertex_count": int(len(vertices_source)),
        "topology_vertex_count": int(len(authority.topology_vertices)),
        "face_count": int(len(faces)),
        "valid_face_count": int(np.count_nonzero(face_valid)),
        "surface_component_count": int(len(set(authority.surface_component_id.tolist()))),
        "sign_candidate_component_count": int(len(authority.sign_component_valid)),
        "sign_valid_component_count": valid_component_count,
        "unsigned_query_count": int(len(query)),
        "signed_query_count": int(len(signed_query_index)),
        "nearest_valid_sign_component_query_count": int(np.count_nonzero(query_nearest_component_is_sign_candidate)),
        "exact_duplicate_collapse_count": int(authority.exact_duplicate_collapse_count),
    }
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "parents": {
            "geometry_npz": {"path": str(geometry_npz), "sha256": sha256(geometry_npz)},
            "source_scene_snapshot": {"path": str(snapshot_json), "sha256": sha256(snapshot_json)},
            "camera_bundle": {"path": str(camera_json), "sha256": sha256(camera_json)},
        },
        "outputs": {
            "topology": {"path": str(topology_path), "sha256": sha256(topology_path)},
            "surface": {"path": str(surface_path), "sha256": sha256(surface_path)},
            "distance_sign": {"path": str(query_path), "sha256": sha256(query_path)},
        },
        "policy": {
            "surface_component_relation": "EDGE_CONNECTED_FACES__ALL_EDGE_INCIDENCE",
            "sign_component_relation": "NONDEGENERATE_FACES_CONNECTED_ONLY_THROUGH_EXACTLY_TWO_REVERSE_ORIENTED_EDGE_INCIDENCE",
            "cross_owner_exact_coincidence_welded": False,
            "unsigned_distance_authority": "NONDEGENERATE_SOURCE_SURFACE",
            "signed_target_storage": "COMPACT_VALID_QUERY_ROWS_ONLY__NO_UNKNOWN_PLACEHOLDER_TARGET",
            "sign_authority": "CLOSED_MANIFOLD_COMPONENT_PLUS_THREE_DIRECTION_PARITY_CONSENSUS__COURT_REQUIRED",
            "union_occupancy_authority": "NOT_MINTED_IN_V1",
            "component_id_semantics": "LOCAL_GEOMETRIC_ADDRESS_ONLY__NOT_MECHANICAL_OR_SEMANTIC_IDENTITY",
        },
        "counts": counts,
        "sign_components": list(authority.sign_components),
        "coverage": {
            "surface_face_fraction": float(np.count_nonzero(face_valid) / max(len(faces), 1)),
            "signed_query_fraction": float(len(signed_query_index) / max(len(query), 1)),
            "nearest_valid_sign_component_query_fraction": float(np.count_nonzero(query_nearest_component_is_sign_candidate) / max(len(query), 1)),
        },
        "authority": {
            "producer_candidate_only": True,
            "independent_court_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
            "source_mutated": False,
        },
        "topology_identity_sha256": authority.identity_sha256,
    }
    manifest["geometry_authority_identity_sha256"] = digest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": STATUS,
        "subject_id": subject_id,
        "surface_component_count": counts["surface_component_count"],
        "sign_valid_component_count": valid_component_count,
        "signed_query_count": counts["signed_query_count"],
        "geometry_authority_identity_sha256": manifest["geometry_authority_identity_sha256"],
    }, sort_keys=True))
    return manifest


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
    build(
        args.subject_id,
        args.geometry_npz,
        args.source_scene_snapshot,
        args.camera_bundle,
        args.output_dir,
        surface_samples=args.surface_samples,
        query_samples=args.query_samples,
    )


if __name__ == "__main__":
    main()
