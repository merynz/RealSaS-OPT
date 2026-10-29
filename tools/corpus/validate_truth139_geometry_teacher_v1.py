#!/usr/bin/env python3
"""Validate Truth139 geometry teacher by rebinding exact source mesh and canonical transform.

The court independently checks that topology arrays come from the exact parent faces, canonical
geometry comes from the bound source->canonical transform, and sampled surface/probe rows reconstruct
from triangle+barycentric addresses.  It does not assume watertightness or mint global SDF sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139GeometryTeacherCourt.v1"
PARENT_SCHEMA = "RealSaS.Truth139GeometryTeacher.v1"
PARENT_STATUS = "PASS_FULL_SOURCE_SURFACE_GEOMETRY_TEACHER_CANDIDATE__DOWNSTREAM_COURTS_REQUIRED"
SNAPSHOT_SCHEMA = "RealSaS.ImportedSourceSceneSnapshot.v2"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _matrix(value) -> np.ndarray:
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-10, rtol=0.0):
        raise ValueError("TRUTH139_GEOMETRY_COURT_TRANSFORM")
    if np.linalg.matrix_rank(m[:3, :3]) != 3:
        raise ValueError("TRUTH139_GEOMETRY_COURT_TRANSFORM_SINGULAR")
    return m


def _points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    h = np.concatenate((p.reshape(-1, 3), np.ones((p.reshape(-1, 3).shape[0], 1))), axis=1)
    out = (h @ matrix.T)[:, :3]
    return out.reshape(p.shape)


def _normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    out = n @ normal_matrix.T
    length = np.linalg.norm(out, axis=1)
    valid = length > 1e-12
    normalized = np.zeros_like(out)
    normalized[valid] = out[valid] / length[valid, None]
    return normalized.reshape(np.asarray(normals).shape)


def _topology(faces: np.ndarray, vertex_count: int):
    f = np.asarray(faces, dtype=np.int64)
    raw = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    raw.sort(axis=1)
    edges, incidence = np.unique(raw, axis=0, return_counts=True)
    boundary = incidence == 1
    nonmanifold = incidence > 2

    parent = np.arange(vertex_count, dtype=np.int64)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    for a, b in edges.tolist():
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(vertex_count)], dtype=np.int64)
    unique = {int(root): idx for idx, root in enumerate(sorted(set(roots.tolist())))}
    vertex_component = np.asarray([unique[int(root)] for root in roots], dtype=np.int32)
    face_component = vertex_component[f[:, 0]]
    if np.any(vertex_component[f] != face_component[:, None]):
        raise ValueError("TRUTH139_GEOMETRY_COURT_COMPONENT_INCONSISTENCY")
    return edges.astype(np.int32), incidence.astype(np.int32), boundary.astype(np.bool_), nonmanifold.astype(np.bool_), vertex_component, face_component.astype(np.int32)


def _face_geometry(vertices: np.ndarray, faces: np.ndarray):
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double) & (double > 1e-12)
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double[valid, None]
    return (0.5 * double).astype(np.float32), normals.astype(np.float32), valid.astype(np.bool_)


def _equal(code: str, got: np.ndarray, expected: np.ndarray) -> None:
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError("TRUTH139_GEOMETRY_COURT_" + code + "_DRIFT")


def _close(code: str, got: np.ndarray, expected: np.ndarray, atol: float) -> None:
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0.0, equal_nan=False):
        raise ValueError("TRUTH139_GEOMETRY_COURT_" + code + "_DRIFT")


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_GEOMETRY_COURT_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_GEOMETRY_COURT_SUBJECT")
    parents = dict(manifest.get("parents") or {})
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_GEOMETRY_COURT_SOURCE_GEOMETRY_DRIFT")
    snapshot_path = _bound_path(dict(parents.get("source_scene_snapshot") or {}), "TRUTH139_GEOMETRY_COURT_SNAPSHOT_DRIFT")
    topology_path = _bound_path(dict(manifest.get("topology_teacher") or {}), "TRUTH139_GEOMETRY_COURT_TOPOLOGY_DRIFT")
    surface_path = _bound_path(dict(manifest.get("surface_teacher") or {}), "TRUTH139_GEOMETRY_COURT_SURFACE_DRIFT")

    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("TRUTH139_GEOMETRY_COURT_SNAPSHOT_SCHEMA")
    matrix = _matrix(snapshot.get("source_world_to_canonical"))
    if str(snapshot.get("source_sha256") or "") != str(parents.get("source_sha256") or ""):
        raise ValueError("TRUTH139_GEOMETRY_COURT_SOURCE_IDENTITY")

    with np.load(geometry_path, allow_pickle=False) as z:
        source_vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float32)
        source_faces = np.asarray(z["faces"], dtype=np.int64)
    if source_vertices.ndim != 2 or source_vertices.shape[1:] != (3,) or source_faces.ndim != 2 or source_faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_GEOMETRY_COURT_SOURCE_LAYOUT")

    expected_area, expected_normal, expected_valid = _face_geometry(source_vertices, source_faces)
    expected_edges, expected_incidence, expected_boundary, expected_nonmanifold, expected_vcomp, expected_fcomp = _topology(source_faces, len(source_vertices))
    expected_canonical = _points(source_vertices, matrix).astype(np.float32)
    expected_normal_canonical = _normals(expected_normal, matrix).astype(np.float32)

    with np.load(topology_path, allow_pickle=False) as z:
        _equal("SOURCE_VERTICES", np.asarray(z["rest_vertices_source_world"], dtype=np.float32), source_vertices)
        _equal("FACES", np.asarray(z["faces"], dtype=np.int64), source_faces)
        _close("CANONICAL_VERTICES", np.asarray(z["rest_vertices_canonical"], dtype=np.float32), expected_canonical, 2e-5)
        _close("FACE_AREA", np.asarray(z["face_area_source_world"], dtype=np.float32), expected_area, 2e-6)
        _close("FACE_NORMAL_SOURCE", np.asarray(z["face_normal_source_world"], dtype=np.float32), expected_normal, 2e-6)
        _close("FACE_NORMAL_CANONICAL", np.asarray(z["face_normal_canonical"], dtype=np.float32), expected_normal_canonical, 2e-5)
        _equal("FACE_VALID", np.asarray(z["nondegenerate_face"], dtype=np.bool_), expected_valid)
        _equal("EDGES", np.asarray(z["unique_edge_index"], dtype=np.int32), expected_edges)
        _equal("EDGE_INCIDENCE", np.asarray(z["edge_face_incidence_count"], dtype=np.int32), expected_incidence)
        _equal("BOUNDARY", np.asarray(z["boundary_edge"], dtype=np.bool_), expected_boundary)
        _equal("NONMANIFOLD", np.asarray(z["nonmanifold_edge"], dtype=np.bool_), expected_nonmanifold)
        _equal("VERTEX_COMPONENT", np.asarray(z["vertex_component_id"], dtype=np.int32), expected_vcomp)
        _equal("FACE_COMPONENT", np.asarray(z["face_component_id"], dtype=np.int32), expected_fcomp)

    with np.load(surface_path, allow_pickle=False) as z:
        triangle = np.asarray(z["triangle_id"], dtype=np.int32).reshape(-1)
        bary = np.asarray(z["barycentric"], dtype=np.float32)
        source_xyz = np.asarray(z["surface_xyz_source_world"], dtype=np.float32)
        canonical_xyz = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
        normal_source = np.asarray(z["normal_source_world"], dtype=np.float32)
        normal_canonical = np.asarray(z["normal_canonical"], dtype=np.float32)
        component = np.asarray(z["component_id"], dtype=np.int32).reshape(-1)
        probe_distance = np.asarray(z["normal_probe_distance_source_world"], dtype=np.float32).reshape(-1)
        plus_source = np.asarray(z["plus_normal_probe_source_world"], dtype=np.float32)
        minus_source = np.asarray(z["minus_normal_probe_source_world"], dtype=np.float32)
        plus_canonical = np.asarray(z["plus_normal_probe_canonical"], dtype=np.float32)
        minus_canonical = np.asarray(z["minus_normal_probe_canonical"], dtype=np.float32)
    n = len(triangle)
    if bary.shape != (n, 3) or source_xyz.shape != canonical_xyz.shape != normal_source.shape != normal_canonical.shape != (n, 3):
        raise ValueError("TRUTH139_GEOMETRY_COURT_SURFACE_LAYOUT")
    if np.any(triangle < 0) or np.any(triangle >= len(source_faces)):
        raise ValueError("TRUTH139_GEOMETRY_COURT_SURFACE_FACE_RANGE")
    if not np.allclose(bary.sum(axis=1), 1.0, atol=1e-6, rtol=0.0) or np.any(bary < -1e-7):
        raise ValueError("TRUTH139_GEOMETRY_COURT_BARYCENTRIC")
    reconstructed = np.sum(source_vertices[source_faces[triangle]] * bary[..., None], axis=1)
    _close("SAMPLE_SOURCE_XYZ", source_xyz, reconstructed.astype(np.float32), 2e-5)
    _close("SAMPLE_CANONICAL_XYZ", canonical_xyz, _points(reconstructed, matrix).astype(np.float32), 3e-5)
    _close("SAMPLE_NORMAL_SOURCE", normal_source, expected_normal[triangle], 2e-6)
    _close("SAMPLE_NORMAL_CANONICAL", normal_canonical, expected_normal_canonical[triangle], 2e-5)
    _equal("SAMPLE_COMPONENT", component, expected_fcomp[triangle])

    if plus_source.shape != minus_source.shape or plus_source.shape != (n, len(probe_distance), 3):
        raise ValueError("TRUTH139_GEOMETRY_COURT_PROBE_LAYOUT")
    expected_plus = source_xyz[:, None, :] + normal_source[:, None, :] * probe_distance[None, :, None]
    expected_minus = source_xyz[:, None, :] - normal_source[:, None, :] * probe_distance[None, :, None]
    _close("PLUS_PROBE_SOURCE", plus_source, expected_plus.astype(np.float32), 3e-5)
    _close("MINUS_PROBE_SOURCE", minus_source, expected_minus.astype(np.float32), 3e-5)
    _close("PLUS_PROBE_CANONICAL", plus_canonical, _points(expected_plus, matrix).astype(np.float32), 5e-5)
    _close("MINUS_PROBE_CANONICAL", minus_canonical, _points(expected_minus, matrix).astype(np.float32), 5e-5)

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_GEOMETRY_TEACHER_REBINDS_SOURCE_MESH_AND_TRANSFORM",
        "subject_id": subject_id,
        "geometry_teacher_identity_sha256": str(manifest.get("geometry_teacher_identity_sha256") or ""),
        "validated_counts": {
            "vertex_count": int(len(source_vertices)),
            "face_count": int(len(source_faces)),
            "surface_sample_count": int(n),
            "component_count": int(len(np.unique(expected_vcomp))),
            "boundary_edge_count": int(np.count_nonzero(expected_boundary)),
            "nonmanifold_edge_count": int(np.count_nonzero(expected_nonmanifold)),
        },
        "policy": {
            "source_mesh_rebound": True,
            "canonical_transform_recomputed": True,
            "topology_recomputed": True,
            "surface_addresses_reconstructed": True,
            "normal_probes_reconstructed": True,
            "global_watertight_sdf_sign_claimed": False,
            "semantic_part_taxonomy_used": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.geometry_teacher_manifest.resolve().read_text())
    result = validate(manifest)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_suffix(out.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(out)
    print(json.dumps({"status": result["status"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
