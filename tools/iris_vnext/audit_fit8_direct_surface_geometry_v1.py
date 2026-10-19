#!/usr/bin/env python3
from __future__ import annotations

"""Independent geometry/topology measurement court for direct IRIS surfaces.

The court consumes only already-materialized model surfaces plus source-defensible
teacher geometry stored in the sealed V4 Fit8 shards. It does not load IRIS, does
not run TESSA, and has no optimization path. V1 intentionally emits measurements
without scientific PASS thresholds; thresholds/promotion policy must be locked in
an independent preregistration before a TESSA A/B decision.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from models.iris.vnext.cohort_v1 import SUBJECT_IDS


SCHEMA = "RealSaS.Fit8DirectIRISSurfaceGeometryCourt.v1"
SUBJECT_SCHEMA = "RealSaS.DirectIRISSurfaceGeometryMeasurement.v1"
SURFACE_SAMPLE_COUNT = 65536
AREA_EPSILON = 1.0e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"DIRECT_SURFACE_COURT_JSON_OBJECT_REQUIRED:{path}")
    return value


def topology_diagnostics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("DIRECT_SURFACE_COURT_MESH_SHAPE_INVALID")
    if len(v) == 0 or len(f) == 0:
        raise ValueError("DIRECT_SURFACE_COURT_EMPTY_MESH")
    if not np.isfinite(v).all():
        raise ValueError("DIRECT_SURFACE_COURT_VERTEX_NONFINITE")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("DIRECT_SURFACE_COURT_FACE_INDEX_INVALID")

    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    degenerate = double_area <= (2.0 * AREA_EPSILON)

    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    nonmanifold_edges = unique_edges[edge_counts > 2]

    parent = np.arange(len(v), dtype=np.int64)
    rank = np.zeros(len(v), dtype=np.uint8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    used = np.unique(f.reshape(-1))
    for a, b in unique_edges:
        union(int(a), int(b))
    component_count = len({find(int(index)) for index in used})

    return {
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "finite_vertices": True,
        "domain_min": [float(x) for x in v.min(axis=0)],
        "domain_max": [float(x) for x in v.max(axis=0)],
        "degenerate_face_count": int(degenerate.sum()),
        "boundary_edge_count": int(len(boundary_edges)),
        "nonmanifold_edge_count": int(len(nonmanifold_edges)),
        "component_count": int(component_count),
        "unique_edge_count": int(len(unique_edges)),
        "area_epsilon": AREA_EPSILON,
    }


def deterministic_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int = SURFACE_SAMPLE_COUNT,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > (2.0 * AREA_EPSILON)
    if not bool(valid.any()):
        raise ValueError("DIRECT_SURFACE_COURT_NO_NONDEGENERATE_FACE")
    valid_face_indices = np.flatnonzero(valid)
    weights = double_area[valid]
    weights = weights / weights.sum()

    rng = np.random.default_rng(int(seed))
    chosen_local = rng.choice(len(valid_face_indices), size=int(count), replace=True, p=weights)
    chosen = valid_face_indices[chosen_local]
    chosen_tri = tri[chosen]
    uv = rng.random((int(count), 2), dtype=np.float64)
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    u = uv[:, 0:1]
    w = uv[:, 1:2]
    points = chosen_tri[:, 0] + u * (chosen_tri[:, 1] - chosen_tri[:, 0]) + w * (chosen_tri[:, 2] - chosen_tri[:, 0])

    normals = cross[chosen]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1.0e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def distance_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("DIRECT_SURFACE_COURT_DISTANCE_INVALID")
    return {
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def measure_surface_against_shard(
    *,
    subject_id: str,
    stage12_path: Path,
    prepared_shard_path: Path,
    teacher_component_count: int | None,
    seed: int,
) -> dict[str, Any]:
    stage = np.load(stage12_path, allow_pickle=False)
    vertices = np.asarray(stage["vertices_normalized"], dtype=np.float32)
    faces = np.asarray(stage["faces"], dtype=np.int64)
    topo = topology_diagnostics(vertices, faces)

    payload = torch.load(prepared_shard_path, map_location="cpu", weights_only=True)
    if str(payload.get("subject_id")) != str(subject_id):
        raise ValueError("DIRECT_SURFACE_COURT_SUBJECT_DRIFT")
    teacher_surface = payload.get("teacher_surface_points_iris")
    mechanical_points = payload.get("mechanical_points")
    mechanical_normals = payload.get("mechanical_normals")
    if not all(isinstance(x, torch.Tensor) for x in (teacher_surface, mechanical_points, mechanical_normals)):
        raise ValueError("DIRECT_SURFACE_COURT_REQUIRED_TEACHER_GEOMETRY_MISSING")
    teacher_surface_np = teacher_surface.float().numpy()
    mech_points_np = mechanical_points.float().numpy()
    mech_normals_np = mechanical_normals.float().numpy()
    if teacher_surface_np.ndim != 2 or teacher_surface_np.shape[1] != 3:
        raise ValueError("DIRECT_SURFACE_COURT_TEACHER_SURFACE_SHAPE_INVALID")
    if mech_points_np.shape != mech_normals_np.shape or mech_points_np.ndim != 2 or mech_points_np.shape[1] != 3:
        raise ValueError("DIRECT_SURFACE_COURT_MECHANICAL_NORMAL_SHAPE_INVALID")
    if not np.isfinite(teacher_surface_np).all() or not np.isfinite(mech_points_np).all() or not np.isfinite(mech_normals_np).all():
        raise ValueError("DIRECT_SURFACE_COURT_TEACHER_GEOMETRY_NONFINITE")

    sample_points, sample_normals = deterministic_surface_samples(vertices, faces, count=SURFACE_SAMPLE_COUNT, seed=seed)
    teacher_tree = cKDTree(teacher_surface_np.astype(np.float64))
    candidate_tree = cKDTree(sample_points.astype(np.float64))

    candidate_to_teacher, _ = teacher_tree.query(sample_points, k=1, workers=-1)
    teacher_to_candidate, _ = candidate_tree.query(teacher_surface_np, k=1, workers=-1)
    mechanical_to_candidate, nearest_index = candidate_tree.query(mech_points_np, k=1, workers=-1)

    candidate_normals = sample_normals[np.asarray(nearest_index, dtype=np.int64)]
    teacher_normal_norm = np.linalg.norm(mech_normals_np, axis=1, keepdims=True).clip(min=1.0e-12)
    teacher_unit = mech_normals_np / teacher_normal_norm
    signed_dot = np.sum(candidate_normals * teacher_unit, axis=1).clip(-1.0, 1.0)
    absolute_dot = np.abs(signed_dot)

    symmetric_chamfer_distance = 0.5 * (float(np.mean(candidate_to_teacher)) + float(np.mean(teacher_to_candidate)))
    symmetric_chamfer_squared = 0.5 * (
        float(np.mean(np.square(candidate_to_teacher))) + float(np.mean(np.square(teacher_to_candidate)))
    )
    teacher_component_error = None
    if teacher_component_count is not None:
        teacher_component_error = int(topo["component_count"] - int(teacher_component_count))

    return {
        "schema": SUBJECT_SCHEMA,
        "subject_id": subject_id,
        "measurement_only": True,
        "scientific_pass_claimed": False,
        "surface_sample_count": SURFACE_SAMPLE_COUNT,
        "candidate_topology": topo,
        "teacher": {
            "surface_point_count": int(len(teacher_surface_np)),
            "mechanical_normal_point_count": int(len(mech_points_np)),
            "component_count": int(teacher_component_count) if teacher_component_count is not None else None,
        },
        "surface_distance": {
            "candidate_to_teacher": distance_summary(candidate_to_teacher),
            "teacher_to_candidate": distance_summary(teacher_to_candidate),
            "symmetric_chamfer_distance": symmetric_chamfer_distance,
            "symmetric_chamfer_squared": symmetric_chamfer_squared,
        },
        "mechanical_surface_distance": distance_summary(mechanical_to_candidate),
        "normal_alignment": {
            "absolute_dot_mean": float(absolute_dot.mean()),
            "absolute_dot_p05": float(np.quantile(absolute_dot, 0.05)),
            "absolute_dot_p50": float(np.quantile(absolute_dot, 0.50)),
            "signed_dot_mean": float(signed_dot.mean()),
        },
        "component_count_error": teacher_component_error,
        "authority": {
            "surface_truth": "SEALED_V4_TEACHER_SURFACE_POINTS_IRIS",
            "normal_truth": "SEALED_V4_MECHANICAL_POINTS_NORMALS",
            "mask_depth_truth": "NOT_PRESENT_IN_V4_SHARD__NOT_MEASURED_BY_THIS_COURT",
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    prepared_root = Path(args.prepared_root).resolve()
    prepared_manifest_path = Path(args.prepared_manifest).resolve()
    surface_root = Path(args.surface_root).resolve()
    output_path = Path(args.output).resolve()

    manifest = load_json(prepared_manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("DIRECT_SURFACE_COURT_PREPARED_MANIFEST_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_COURT_SUBJECT_ORDER_DRIFT")

    batch_receipt_path = surface_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    batch_receipt = load_json(batch_receipt_path)
    if batch_receipt.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1":
        raise ValueError("DIRECT_SURFACE_COURT_BATCH_SCHEMA_INVALID")
    if batch_receipt.get("status") != "MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING":
        raise ValueError("DIRECT_SURFACE_COURT_BATCH_NOT_MATERIALIZED")
    if batch_receipt.get("tessa_invoked") is not False or batch_receipt.get("optimizer_steps_executed") != 0:
        raise ValueError("DIRECT_SURFACE_COURT_A_ARM_CONTAMINATED")
    receipt_rows = list(batch_receipt.get("subjects") or ())
    if tuple(row.get("subject_id") for row in receipt_rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_COURT_BATCH_SUBJECT_ORDER_DRIFT")

    results: list[dict[str, Any]] = []
    for subject_index, (manifest_row, receipt_row) in enumerate(zip(rows, receipt_rows, strict=True)):
        subject_id = SUBJECT_IDS[subject_index]
        shard = (prepared_root / str(manifest_row["shard_file"])).resolve()
        if not shard.is_relative_to(prepared_root) or sha256_file(shard) != str(manifest_row["shard_sha256"]):
            raise ValueError("DIRECT_SURFACE_COURT_PREPARED_SHARD_DRIFT:" + subject_id)
        subject_root = surface_root / "subjects" / subject_id
        stage12 = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
        receipt = load_json(subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json")
        if sha256_file(stage12) != str(receipt_row["stage12_npz_sha256"]):
            raise ValueError("DIRECT_SURFACE_COURT_STAGE12_HASH_DRIFT:" + subject_id)
        if receipt.get("candidate_binding_hash") != receipt_row.get("candidate_binding_hash"):
            raise ValueError("DIRECT_SURFACE_COURT_CANDIDATE_BINDING_DRIFT:" + subject_id)
        if receipt.get("compiler_qualified") is not False or receipt.get("scientific_pass_claimed") is not False:
            raise ValueError("DIRECT_SURFACE_COURT_PREQUALIFIED_INPUT_FORBIDDEN")

        metadata = manifest_row.get("metadata") or {}
        topology_teacher = metadata.get("topology_teacher") or {}
        teacher_components = topology_teacher.get("component_count")
        result = measure_surface_against_shard(
            subject_id=subject_id,
            stage12_path=stage12,
            prepared_shard_path=shard,
            teacher_component_count=int(teacher_components) if teacher_components is not None else None,
            seed=26091991 + subject_index,
        )
        result["candidate_binding_hash"] = receipt_row["candidate_binding_hash"]
        result["stage12_npz_sha256"] = receipt_row["stage12_npz_sha256"]
        results.append(result)

    summary = {
        "schema": SCHEMA,
        "status": "MEASURED__THRESHOLDS_NOT_PREREGISTERED__NO_PASS_CLAIM",
        "surface_batch_receipt_sha256": sha256_file(batch_receipt_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": results,
        "surface_sample_count_per_subject": SURFACE_SAMPLE_COUNT,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "mask_depth_court_executed": False,
        "scientific_pass_claimed": False,
        "next_action": "PREREGISTER_GEOMETRY_THRESHOLDS_AND_BIND_STAGE01_RENDER_AUTHORITY_BEFORE_PROMOTION_DECISION",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--surface-root", required=True)
    p.add_argument("--output", required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    result = audit(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
