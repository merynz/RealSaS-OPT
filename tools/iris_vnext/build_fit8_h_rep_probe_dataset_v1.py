#!/usr/bin/env python3
from __future__ import annotations

"""Build deterministic pointwise geometry-probe truth from sealed V4 Fit8 shards.

This is a representation-diagnostic dataset. It uses the same sealed teacher
surface points and mechanical points/normals already consumed by independent
surface courts. No authored topology, rig, skin, TESSA output, or A1 surface is
used as a probe input. Targets are explicitly point-cloud/mechanical proxies and
are never re-described as exact SDF/occupancy truth.
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
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import load_json, sha256_file

SCHEMA = "RealSaS.HRepP0ProbeDataset.v1"
SUBJECT_SCHEMA = "RealSaS.HRepP0ProbeSubject.v1"
OFFSETS = np.asarray([0.0, 0.015, -0.015, 0.03, -0.03, 0.06, -0.06, 0.10, -0.10], dtype=np.float32)
NEAR_THRESHOLD_IRIS = 0.03
NORMAL_VALID_DISTANCE_IRIS = 0.08


def _tensor(payload: dict[str, Any], key: str, subject_id: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"H_REP_P0_REQUIRED_TENSOR_MISSING:{subject_id}:{key}")
    array = value.detach().cpu().float().numpy()
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"H_REP_P0_TENSOR_INVALID:{subject_id}:{key}")
    return array.astype(np.float32, copy=False)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value.astype(np.float64), axis=1, keepdims=True)
    if np.any(norm <= 1e-12) or not np.isfinite(norm).all():
        raise ValueError("H_REP_P0_NORMAL_INVALID")
    return (value / norm).astype(np.float32)


def _make_split(*, rng: np.random.Generator, mechanical_points: np.ndarray, mechanical_normals: np.ndarray, reference_points: np.ndarray, count_near: int, count_uniform: int) -> dict[str, np.ndarray]:
    normals = _normalize_rows(mechanical_normals)
    queries: list[np.ndarray] = []
    while sum(len(x) for x in queries) < count_near:
        remaining = count_near - sum(len(x) for x in queries)
        draw = max(remaining * 2, 1024)
        idx = rng.integers(0, len(mechanical_points), size=draw, endpoint=False)
        offset = OFFSETS[rng.integers(0, len(OFFSETS), size=draw, endpoint=False)]
        candidate = mechanical_points[idx] + normals[idx] * offset[:, None]
        valid = np.max(np.abs(candidate), axis=1) <= 1.0
        if valid.any():
            queries.append(candidate[valid][:remaining].astype(np.float32))
    near_query = np.concatenate(queries, axis=0)[:count_near]
    uniform_query = rng.uniform(-1.0, 1.0, size=(count_uniform, 3)).astype(np.float32)
    query = np.concatenate((near_query, uniform_query), axis=0)
    order = rng.permutation(len(query))
    query = query[order]

    ref_tree = cKDTree(reference_points.astype(np.float64))
    distance, nearest_index = ref_tree.query(query.astype(np.float64), k=1, workers=-1)
    nearest_reference = reference_points[np.asarray(nearest_index, dtype=np.int64)]
    offset_to_reference = nearest_reference - query

    normal_tree = cKDTree(mechanical_points.astype(np.float64))
    normal_distance, normal_index = normal_tree.query(query.astype(np.float64), k=1, workers=-1)
    normal_target = normals[np.asarray(normal_index, dtype=np.int64)]
    normal_valid = normal_distance <= NORMAL_VALID_DISTANCE_IRIS
    near_label = distance <= NEAR_THRESHOLD_IRIS

    return {
        "xyz_iris": query.astype(np.float32),
        "nearest_reference_iris": nearest_reference.astype(np.float32),
        "distance_iris": np.asarray(distance, dtype=np.float32),
        "offset_to_reference_iris": offset_to_reference.astype(np.float32),
        "normal_unit": normal_target.astype(np.float32),
        "normal_valid": np.asarray(normal_valid, dtype=np.uint8),
        "near_label": np.asarray(near_label, dtype=np.uint8),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    binding = load_json(Path(args.binding).resolve())
    if binding.get("schema") != "RealSaS.HRepP0InputBinding.v1":
        raise ValueError("H_REP_P0_BINDING_SCHEMA_INVALID")
    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("H_REP_P0_MANIFEST_SCHEMA_INVALID")
    if sha256_file(manifest_path) != binding["prepared"]["manifest"]["sha256"]:
        raise ValueError("H_REP_P0_MANIFEST_SHA_DRIFT")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("H_REP_P0_SUBJECT_ORDER_DRIFT")

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("H_REP_P0_DATASET_OUTPUT_DIR_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(args.prepared_root).resolve()
    result_rows = []
    for subject_index, row in enumerate(rows):
        subject_id = str(row["subject_id"])
        shard_path = (prepared_root / str(row["shard_file"])).resolve()
        if not shard_path.is_relative_to(prepared_root) or sha256_file(shard_path) != str(row["shard_sha256"]):
            raise ValueError("H_REP_P0_SHARD_SHA_DRIFT:" + subject_id)
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        teacher_surface = _tensor(payload, "teacher_surface_points_iris", subject_id)
        mechanical_points = _tensor(payload, "mechanical_points", subject_id)
        mechanical_normals = _tensor(payload, "mechanical_normals", subject_id)
        if mechanical_points.shape != mechanical_normals.shape:
            raise ValueError("H_REP_P0_MECHANICAL_SHAPE_DRIFT:" + subject_id)
        reference_points = np.concatenate((teacher_surface, mechanical_points), axis=0)
        train = _make_split(
            rng=np.random.default_rng(26101980 + subject_index),
            mechanical_points=mechanical_points,
            mechanical_normals=mechanical_normals,
            reference_points=reference_points,
            count_near=int(args.train_near),
            count_uniform=int(args.train_uniform),
        )
        holdout = _make_split(
            rng=np.random.default_rng(26102980 + subject_index),
            mechanical_points=mechanical_points,
            mechanical_normals=mechanical_normals,
            reference_points=reference_points,
            count_near=int(args.holdout_near),
            count_uniform=int(args.holdout_uniform),
        )
        out_path = output_root / f"{subject_id}.h_rep_p0_probe.npz"
        np.savez_compressed(out_path, **{f"train_{k}": v for k,v in train.items()}, **{f"holdout_{k}": v for k,v in holdout.items()})
        row_out = {
            "schema": SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "prepared_shard_sha256": str(row["shard_sha256"]),
            "payload_file": out_path.name,
            "payload_sha256": sha256_file(out_path),
            "teacher_surface_point_count": int(len(teacher_surface)),
            "mechanical_point_count": int(len(mechanical_points)),
            "train_count": int(len(train["xyz_iris"])),
            "holdout_count": int(len(holdout["xyz_iris"])),
            "train_near_positive_fraction": float(train["near_label"].mean()),
            "holdout_near_positive_fraction": float(holdout["near_label"].mean()),
        }
        result_rows.append(row_out)
        print(f"H_REP_P0_DATASET {subject_id} train={row_out['train_count']} holdout={row_out['holdout_count']} near={row_out['holdout_near_positive_fraction']:.4f}", flush=True)

    result = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_H_REP_P0_POINT_PROXY_DATASET",
        "input_binding_sha256": sha256_file(Path(args.binding).resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": result_rows,
        "target_authority": {
            "distance_and_offset": "NEAREST_POINT_TO_SEALED_V4_TEACHER_SURFACE_PLUS_MECHANICAL_POINT_SET__POINT_CLOUD_PROXY_NOT_EXACT_SDF",
            "normal": "NEAREST_SEALED_V4_MECHANICAL_POINT_NORMAL_WITH_DISTANCE_VALIDITY",
            "near_label": f"POINT_CLOUD_DISTANCE_LE_{NEAR_THRESHOLD_IRIS}_IRIS",
        },
        "teacher_truth_in_probe_input": False,
        "mesh_extraction_invoked": False,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
        "scientific_pass_claimed": False,
        "next_action": "TRAIN_EQUAL_CAPACITY_H_REP_P0_CONTROL_PROBES",
    }
    receipt = output_root / "H_REP_P0_PROBE_DATASET_RECEIPT.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", default="canonical/H_REP_P0_INPUT_BINDING_V1.json")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-near", type=int, default=24576)
    p.add_argument("--train-uniform", type=int, default=8192)
    p.add_argument("--holdout-near", type=int, default=6144)
    p.add_argument("--holdout-uniform", type=int, default=2048)
    result=build(p.parse_args()); print(json.dumps({"status":result["status"],"subjects":len(result["subjects"])},sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
