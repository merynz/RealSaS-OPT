#!/usr/bin/env python3
"""Build full-source-surface Truth139 geometry teacher truth.

The current IRIS F* field is deliberately image-derived.  This producer preserves the privileged
source-mesh truth separately so later experiments can use exact surface/topology supervision without
silently redefining F* or assuming that arbitrary character meshes are closed, consistently wound
solids with a globally valid SDF sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256
from tools.corpus.build_truth139_dense_graphics_teacher_v1 import transform_normals, transform_points

SCHEMA = "RealSaS.Truth139GeometryTeacher.v1"
SNAPSHOT_SCHEMA = "RealSaS.ImportedSourceSceneSnapshot.v2"
CONTRACT = "canonical/TRUTH139_GEOMETRY_TEACHER_V1.json"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _matrix(value, code: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(code)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10, rtol=0.0):
        raise ValueError(code + "_AFFINE")
    if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
        raise ValueError(code + "_SINGULAR")
    return matrix


def _load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_VERTEX_LAYOUT")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_FACE_LAYOUT")
    return vertices, faces


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-12)
    normal = np.zeros_like(cross)
    normal[valid] = cross[valid] / double_area[valid, None]
    return (0.5 * double_area).astype(np.float32), normal.astype(np.float32), valid.astype(np.bool_)


def _topology(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    raw_edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    raw_edges.sort(axis=1)
    edges, incidence = np.unique(raw_edges, axis=0, return_counts=True)
    boundary = incidence == 1
    nonmanifold = incidence > 2

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception as exc:  # pragma: no cover - worker dependency
        raise RuntimeError("TRUTH139_GEOMETRY_TEACHER_SCIPY_REQUIRED") from exc
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.ones(len(row), dtype=np.uint8)
    adjacency = coo_matrix((data, (row, col)), shape=(len(vertices), len(vertices))).tocsr()
    component_count, vertex_component = connected_components(adjacency, directed=False, return_labels=True)
    face_component = vertex_component[f[:, 0]]
    if np.any(vertex_component[f] != face_component[:, None]):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_COMPONENT_INCONSISTENCY")
    return (
        edges.astype(np.int32),
        incidence.astype(np.int32),
        boundary.astype(np.bool_),
        nonmanifold.astype(np.bool_),
        vertex_component.astype(np.int32),
        face_component.astype(np.int32),
    )


def _sample_surface(
    *,
    subject_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_area: np.ndarray,
    face_normal: np.ndarray,
    face_valid: np.ndarray,
    face_component: np.ndarray,
    maximum_samples: int,
) -> dict[str, np.ndarray]:
    if maximum_samples < 1:
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_SAMPLE_BUDGET")
    valid_faces = np.flatnonzero(face_valid)
    if not len(valid_faces):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_ALL_FACES_DEGENERATE")
    components = np.unique(face_component[valid_faces])
    if len(components) > maximum_samples:
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_COMPONENTS_EXCEED_SAMPLE_BUDGET")
    seed = int(hashlib.sha256(f"{subject_id}|full_surface_geometry".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)

    selected_faces: list[int] = []
    selected_bary: list[list[float]] = []
    representative = []
    for component in components:
        ids = valid_faces[face_component[valid_faces] == component]
        face = int(ids[np.argmax(face_area[ids])])
        representative.append(face)
        selected_faces.append(face)
        selected_bary.append([1.0 / 3.0] * 3)

    represented = set(representative)
    remaining_budget = maximum_samples - len(selected_faces)
    if remaining_budget > 0:
        remaining_faces = np.asarray([int(i) for i in valid_faces if int(i) not in represented], dtype=np.int64)
        take_unique = min(len(remaining_faces), remaining_budget)
        if take_unique:
            probabilities = face_area[remaining_faces].astype(np.float64)
            probabilities /= probabilities.sum()
            chosen = rng.choice(remaining_faces, size=take_unique, replace=False, p=probabilities)
            for face in np.sort(chosen):
                selected_faces.append(int(face))
                selected_bary.append([1.0 / 3.0] * 3)
        remaining_budget -= take_unique

    if remaining_budget > 0:
        probabilities = face_area[valid_faces].astype(np.float64)
        probabilities /= probabilities.sum()
        extra = rng.choice(valid_faces, size=remaining_budget, replace=True, p=probabilities)
        r1 = rng.random(remaining_budget)
        r2 = rng.random(remaining_budget)
        root = np.sqrt(r1)
        bary = np.stack((1.0 - root, root * (1.0 - r2), root * r2), axis=1)
        selected_faces.extend(int(x) for x in extra)
        selected_bary.extend(bary.tolist())

    triangle = np.asarray(selected_faces, dtype=np.int64)
    barycentric = np.asarray(selected_bary, dtype=np.float64)
    if triangle.shape != (maximum_samples,) or barycentric.shape != (maximum_samples, 3):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_SAMPLE_COUNT_DRIFT")
    if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1.0e-7, rtol=0.0) or np.any(barycentric < 0.0):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_BARYCENTRIC_INVALID")
    point = np.sum(vertices[faces[triangle]] * barycentric[..., None], axis=1)
    return {
        "triangle_id": triangle.astype(np.int32),
        "barycentric": barycentric.astype(np.float32),
        "surface_xyz_source_world": point.astype(np.float32),
        "normal_source_world": face_normal[triangle].astype(np.float32),
        "face_area_source_world": face_area[triangle].astype(np.float32),
        "component_id": face_component[triangle].astype(np.int32),
    }


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(
    *,
    subject_id: str,
    geometry_npz: Path,
    source_scene_snapshot: Path,
    output_dir: Path,
    maximum_surface_samples: int = 131072,
    normal_probe_offsets_relative: tuple[float, ...] = (0.0005, 0.002, 0.01),
) -> dict:
    geometry_path = geometry_npz.resolve()
    snapshot_path = source_scene_snapshot.resolve()
    output_dir = output_dir.resolve()
    if not subject_id or not geometry_path.is_file() or not snapshot_path.is_file():
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_INPUT")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_SNAPSHOT_SCHEMA")
    source_to_canonical = _matrix(snapshot.get("source_world_to_canonical"), "TRUTH139_GEOMETRY_TEACHER_TRANSFORM")
    vertices, faces = _load_geometry(geometry_path)
    vertices_canonical = transform_points(vertices, source_to_canonical)
    face_area, face_normal, face_valid = _face_geometry(vertices, faces)
    face_normal_canonical = transform_normals(face_normal, source_to_canonical)
    edges, edge_incidence, boundary_edge, nonmanifold_edge, vertex_component, face_component = _topology(vertices, faces)
    source_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    canonical_diagonal = float(np.linalg.norm(vertices_canonical.max(axis=0) - vertices_canonical.min(axis=0)))
    if not math.isfinite(source_diagonal) or source_diagonal <= 0 or not math.isfinite(canonical_diagonal) or canonical_diagonal <= 0:
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_DEGENERATE_BOUNDS")
    offsets = np.asarray(normal_probe_offsets_relative, dtype=np.float64)
    if offsets.ndim != 1 or not len(offsets) or np.any(~np.isfinite(offsets)) or np.any(offsets <= 0):
        raise ValueError("TRUTH139_GEOMETRY_TEACHER_PROBE_OFFSETS")

    output_dir.mkdir(parents=True, exist_ok=True)
    topology_path = output_dir / "GEOMETRY_TOPOLOGY_TEACHER.npz"
    _write_npz(
        topology_path,
        rest_vertices_source_world=vertices.astype(np.float32),
        rest_vertices_canonical=vertices_canonical.astype(np.float32),
        faces=faces.astype(np.int32),
        face_area_source_world=face_area,
        face_normal_source_world=face_normal,
        face_normal_canonical=face_normal_canonical,
        nondegenerate_face=face_valid,
        unique_edge_index=edges,
        edge_face_incidence_count=edge_incidence,
        boundary_edge=boundary_edge,
        nonmanifold_edge=nonmanifold_edge,
        vertex_component_id=vertex_component,
        face_component_id=face_component,
    )

    samples = _sample_surface(
        subject_id=subject_id,
        vertices=vertices,
        faces=faces,
        face_area=face_area,
        face_normal=face_normal,
        face_valid=face_valid,
        face_component=face_component,
        maximum_samples=int(maximum_surface_samples),
    )
    samples["surface_xyz_canonical"] = transform_points(samples["surface_xyz_source_world"], source_to_canonical)
    samples["normal_canonical"] = transform_normals(samples["normal_source_world"], source_to_canonical)
    probe_distance = (offsets * source_diagonal).astype(np.float32)
    samples["normal_probe_distance_source_world"] = probe_distance
    samples["plus_normal_probe_source_world"] = (
        samples["surface_xyz_source_world"][:, None, :] + samples["normal_source_world"][:, None, :] * probe_distance[None, :, None]
    ).astype(np.float32)
    samples["minus_normal_probe_source_world"] = (
        samples["surface_xyz_source_world"][:, None, :] - samples["normal_source_world"][:, None, :] * probe_distance[None, :, None]
    ).astype(np.float32)
    samples["plus_normal_probe_canonical"] = transform_points(
        samples["plus_normal_probe_source_world"].reshape(-1, 3), source_to_canonical
    ).reshape(len(samples["triangle_id"]), len(offsets), 3)
    samples["minus_normal_probe_canonical"] = transform_points(
        samples["minus_normal_probe_source_world"].reshape(-1, 3), source_to_canonical
    ).reshape(len(samples["triangle_id"]), len(offsets), 3)
    surface_path = output_dir / "GEOMETRY_SURFACE_SAMPLES.npz"
    _write_npz(surface_path, **samples)

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_FULL_SOURCE_SURFACE_GEOMETRY_TEACHER_CANDIDATE__DOWNSTREAM_COURTS_REQUIRED",
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "source_scene_snapshot": {"path": str(snapshot_path), "sha256": sha256(snapshot_path)},
            "source_sha256": str(snapshot.get("source_sha256") or ""),
        },
        "topology_teacher": {"path": str(topology_path), "sha256": sha256(topology_path)},
        "surface_teacher": {"path": str(surface_path), "sha256": sha256(surface_path)},
        "counts": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "nondegenerate_face_count": int(np.count_nonzero(face_valid)),
            "degenerate_face_count": int(np.count_nonzero(~face_valid)),
            "unique_edge_count": int(len(edges)),
            "boundary_edge_count": int(np.count_nonzero(boundary_edge)),
            "nonmanifold_edge_count": int(np.count_nonzero(nonmanifold_edge)),
            "connected_component_count": int(len(np.unique(vertex_component))),
            "surface_sample_count": int(len(samples["triangle_id"])),
        },
        "bounds": {
            "source_bbox_min": vertices.min(axis=0).astype(float).tolist(),
            "source_bbox_max": vertices.max(axis=0).astype(float).tolist(),
            "source_bbox_diagonal": source_diagonal,
            "canonical_bbox_min": vertices_canonical.min(axis=0).astype(float).tolist(),
            "canonical_bbox_max": vertices_canonical.max(axis=0).astype(float).tolist(),
            "canonical_bbox_diagonal": canonical_diagonal,
        },
        "sampling": {
            "policy": "DETERMINISTIC_COMPONENT_AWARE_AREA_STRATIFIED",
            "maximum_surface_samples": int(maximum_surface_samples),
            "normal_probe_offsets_relative_to_source_bbox_diagonal": offsets.tolist(),
            "local_orientation_probes_are_global_sdf_sign": False,
        },
        "authority": {
            "full_source_surface_preserved": True,
            "image_derived_fstar_redefined": False,
            "global_watertight_sdf_claimed": False,
            "mesh_winding_used_as_global_inside_outside_truth": False,
            "semantic_part_taxonomy_used": False,
            "training_performed": False,
        },
        "qualification_minted": False,
    }
    payload["geometry_teacher_identity_sha256"] = _digest(payload)
    manifest_path = output_dir / "GEOMETRY_TEACHER_MANIFEST.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "subject_id": subject_id,
                "surface_sample_count": payload["counts"]["surface_sample_count"],
                "component_count": payload["counts"]["connected_component_count"],
                "geometry_teacher_identity_sha256": payload["geometry_teacher_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--source-scene-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-surface-samples", type=int, default=131072)
    args = parser.parse_args()
    build(
        subject_id=args.subject_id,
        geometry_npz=args.geometry_npz,
        source_scene_snapshot=args.source_scene_snapshot,
        output_dir=args.output_dir,
        maximum_surface_samples=args.maximum_surface_samples,
    )


if __name__ == "__main__":
    main()
