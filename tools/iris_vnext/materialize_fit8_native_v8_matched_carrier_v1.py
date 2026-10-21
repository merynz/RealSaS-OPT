#!/usr/bin/env python3
from __future__ import annotations

"""Project exact V8 native-R24 FlexiCubes witnesses into the legacy Stage12
consumer envelope without changing carrier geometry.

This is an H1 experiment compatibility adapter only. It does not decode a raw
scalar field, does not run T512/T1024 marching tetrahedra, and does not upgrade
the native witness to product/high-resolution/compiler-qualified authority.

The produced Stage12 files preserve the native witness vertices/faces exactly;
`vertex_normals` is renamed to the legacy `implicit_normals` slot solely because
existing TESSA corpus consumers require one per-vertex normal array.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS

NATIVE_BATCH_SCHEMA = "RealSaS.Fit8V8NativeFlexiCubesMaterialization.v1"
NATIVE_SUBJECT_SCHEMA = "RealSaS.V8NativeFlexiCubesWitness.v1"
DIRECT_COMPAT_BATCH_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1"
DIRECT_COMPAT_SUBJECT_SCHEMA = "RealSaS.DirectIRISSurfaceReceipt.v1"
CARRIER_ID = "RealSaS.FixedSurfaceCarrier.ExactV8NativeFlexiCubesR24.v1"
ADAPTER_ID = "RealSaS.NativeV8MatchedCarrierStage12CompatibilityProjection.v1"
CARRIER_AUTHORITY = "INFERENCE_ADMISSIBLE_FIXED_SURFACE_CARRIER"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("utf-8"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("utf-8"))
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NATIVE_MATCHED_CARRIER_JSON_OBJECT_REQUIRED:" + str(path))
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_head(repo_root: Path) -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if git_head(repo_root) != expected_source_sha:
        raise RuntimeError("NATIVE_MATCHED_CARRIER_CHECKOUT_SHA_DRIFT")

    native_root = Path(args.native_root).resolve()
    native_batch_path = native_root / "FIT8_V8_NATIVE_FLEXICUBES_BATCH_RECEIPT.json"
    native_batch = load_json(native_batch_path)
    if native_batch.get("schema") != NATIVE_BATCH_SCHEMA:
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_BATCH_SCHEMA_INVALID")
    if native_batch.get("status") != "MATERIALIZED_NATIVE_V8_FLEXICUBES_WITNESSES__MEASUREMENT_ONLY":
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_BATCH_STATUS_INVALID")
    if native_batch.get("execution_source_sha") != expected_source_sha:
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SOURCE_SHA_DRIFT")
    if tuple(native_batch.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SUBJECT_ORDER_DRIFT")
    if native_batch.get("high_resolution_capability_claimed") is not False:
        raise ValueError("NATIVE_MATCHED_CARRIER_ILLEGAL_HIGH_RES_AUTHORITY")
    if native_batch.get("tessa_invoked") is not False or int(native_batch.get("optimizer_steps_executed", -1)) != 0:
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_CONTAMINATED")

    prepared_manifest_path = Path(args.prepared_manifest).resolve()
    prepared = load_json(prepared_manifest_path)
    if prepared.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("NATIVE_MATCHED_CARRIER_PREPARED_SCHEMA_INVALID")
    prepared_rows = list(prepared.get("subjects") or ())
    if tuple(row.get("subject_id") for row in prepared_rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_MATCHED_CARRIER_PREPARED_SUBJECT_ORDER_DRIFT")
    if sha256_file(prepared_manifest_path) != native_batch.get("prepared_manifest_sha256"):
        raise ValueError("NATIVE_MATCHED_CARRIER_PREPARED_MANIFEST_SHA_DRIFT")
    prepared_by_subject = {str(row["subject_id"]): row for row in prepared_rows}

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("NATIVE_MATCHED_CARRIER_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    native_rows = {str(row["subject_id"]): row for row in native_batch.get("subjects") or ()}
    if tuple(native_rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_ROW_ORDER_DRIFT")

    batch_rows: list[dict[str, Any]] = []
    first_v8_authority: dict[str, Any] | None = None
    for subject_id in SUBJECT_IDS:
        native_subject_root = native_root / "subjects" / subject_id
        native_receipt_path = native_subject_root / "V8_NATIVE_FLEXICUBES_R24_RECEIPT.json"
        native_witness_path = native_subject_root / "V8_NATIVE_FLEXICUBES_R24.npz"
        native_receipt = load_json(native_receipt_path)
        native_row = native_rows[subject_id]
        if native_receipt.get("schema") != NATIVE_SUBJECT_SCHEMA:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SUBJECT_SCHEMA_INVALID:" + subject_id)
        if native_receipt.get("subject_id") != subject_id:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SUBJECT_ID_DRIFT:" + subject_id)
        if native_receipt.get("execution_source_sha") != expected_source_sha:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SUBJECT_SOURCE_DRIFT:" + subject_id)
        if int(native_receipt.get("resolution", -1)) != 24:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_RESOLUTION_DRIFT:" + subject_id)
        if native_receipt.get("surface_semantics") != "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_TRAINING_RESOLUTION":
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SEMANTICS_DRIFT:" + subject_id)
        if native_receipt.get("teacher_truth_used_in_inference_forward") is not False:
            raise ValueError("NATIVE_MATCHED_CARRIER_TEACHER_LEAK:" + subject_id)
        if native_receipt.get("high_resolution_capability_claimed") is not False:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_SUBJECT_HIGH_RES_CLAIM:" + subject_id)
        if sha256_file(native_witness_path) != native_receipt.get("witness_sha256"):
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_WITNESS_SHA_DRIFT:" + subject_id)
        if native_row.get("witness_sha256") != native_receipt.get("witness_sha256"):
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_BATCH_WITNESS_DRIFT:" + subject_id)
        if sha256_file(native_receipt_path) != native_row.get("receipt_sha256"):
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_RECEIPT_SHA_DRIFT:" + subject_id)
        v8_authority = dict(native_receipt.get("v8_authority") or {})
        if first_v8_authority is None:
            first_v8_authority = v8_authority
        elif v8_authority != first_v8_authority:
            raise ValueError("NATIVE_MATCHED_CARRIER_V8_AUTHORITY_DRIFT:" + subject_id)

        with np.load(native_witness_path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices_normalized"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.int32)
            normals = np.asarray(data["vertex_normals"], dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_VERTEX_SHAPE_INVALID:" + subject_id)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(vertices) == 0 or len(faces) == 0:
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_FACE_SHAPE_INVALID:" + subject_id)
        if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_NONFINITE:" + subject_id)
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("NATIVE_MATCHED_CARRIER_NATIVE_FACE_INDEX_INVALID:" + subject_id)

        native_vertices_sha = array_sha256(vertices)
        native_faces_sha = array_sha256(faces)
        native_normals_sha = array_sha256(normals)

        subject_root = output_root / "subjects" / subject_id
        subject_root.mkdir(parents=True, exist_ok=True)
        stage_path = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
        np.savez_compressed(
            stage_path,
            vertices_normalized=vertices,
            faces=faces,
            implicit_normals=normals,
        )
        stage_sha = sha256_file(stage_path)
        with np.load(stage_path, allow_pickle=False) as projected:
            if array_sha256(np.asarray(projected["vertices_normalized"], dtype=np.float32)) != native_vertices_sha:
                raise RuntimeError("NATIVE_MATCHED_CARRIER_VERTEX_MUTATION:" + subject_id)
            if array_sha256(np.asarray(projected["faces"], dtype=np.int32)) != native_faces_sha:
                raise RuntimeError("NATIVE_MATCHED_CARRIER_FACE_MUTATION:" + subject_id)
            if array_sha256(np.asarray(projected["implicit_normals"], dtype=np.float32)) != native_normals_sha:
                raise RuntimeError("NATIVE_MATCHED_CARRIER_NORMAL_MUTATION:" + subject_id)

        candidate_binding = {
            "schema": "RealSaS.FixedSurfaceCarrierCandidateBinding.v1",
            "carrier_id": CARRIER_ID,
            "carrier_authority": CARRIER_AUTHORITY,
            "subject_id": subject_id,
            "execution_source_sha": expected_source_sha,
            "native_witness_sha256": native_receipt["witness_sha256"],
            "native_receipt_sha256": sha256_file(native_receipt_path),
            "vertices_array_sha256": native_vertices_sha,
            "faces_array_sha256": native_faces_sha,
            "normals_array_sha256": native_normals_sha,
            "stage12_compatibility_sha256": stage_sha,
            "geometry_mutated": False,
            "topology_mutated": False,
            "normal_values_mutated": False,
        }
        candidate_hash = value_sha256(candidate_binding)
        prepared_row = prepared_by_subject[subject_id]
        receipt = {
            "schema": DIRECT_COMPAT_SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "source_commit": expected_source_sha,
            "checkpoint_sha256": v8_authority.get("checkpoint_sha256"),
            "scene_binding_sha256": native_receipt.get("scene_binding_sha256"),
            "candidate_binding_hash": candidate_hash,
            "stage12_npz_sha256": stage_sha,
            "decoder_id": CARRIER_ID,
            "base_cells": 24,
            "fine_cells": 24,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "field_coordinate_transform": "NATIVE_V8_IRIS_NORMALIZED_DOMAIN",
            "field_scalar_transform": "NOT_APPLICABLE__NATIVE_TRAINED_EXTRACTION_PATH",
            "teacher_truth_used": False,
            "compiler_qualified": False,
            "training_authorized": False,
            "scientific_pass_claimed": False,
            "status": "FIXED_NATIVE_V8_R24_CARRIER_PROJECTED__NO_AUTHORITY_UPGRADE",
            "carrier_authority": CARRIER_AUTHORITY,
            "surface_semantics": "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_R24",
            "native_witness_file": native_witness_path.name,
            "native_witness_sha256": native_receipt["witness_sha256"],
            "native_receipt_sha256": sha256_file(native_receipt_path),
            "compatibility_adapter_id": ADAPTER_ID,
            "compatibility_projection_only": True,
            "geometry_mutated": False,
            "topology_mutated": False,
            "normal_slot_rename_only": True,
            "high_resolution_capability_claimed": False,
            "product_surface_claimed": False,
            "raw_field_sparse_t512_t1024_invoked": False,
        }
        receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
        write_json(receipt_path, receipt)

        batch_rows.append({
            "subject_id": subject_id,
            "prepared_shard_file": str(prepared_row["shard_file"]),
            "prepared_shard_sha256": str(prepared_row["shard_sha256"]),
            "scene_binding_sha256": native_receipt.get("scene_binding_sha256"),
            "scene_planes_sha256": native_receipt.get("scene_planes_sha256"),
            "candidate_binding_hash": candidate_hash,
            "stage12_npz_sha256": stage_sha,
            "receipt_sha256": sha256_file(receipt_path),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "field_query_api": "NOT_USED__NATIVE_V8_TRAINED_EXTRACTION",
            "compiler_qualified": False,
            "carrier_authority": CARRIER_AUTHORITY,
            "native_witness_sha256": native_receipt["witness_sha256"],
        })

    batch = {
        "schema": DIRECT_COMPAT_BATCH_SCHEMA,
        "status": "FIXED_NATIVE_V8_R24_CARRIER_PROJECTED_FOR_MATCHED_H1_AB__NO_AUTHORITY_UPGRADE",
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "native_batch_receipt_sha256": sha256_file(native_batch_path),
        "carrier_id": CARRIER_ID,
        "carrier_authority": CARRIER_AUTHORITY,
        "surface_semantics": "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_R24",
        "compatibility_adapter_id": ADAPTER_ID,
        "compatibility_projection_only": True,
        "geometry_mutated": False,
        "topology_mutated": False,
        "normal_slot_rename_only": True,
        "subject_order": list(SUBJECT_IDS),
        "subjects": batch_rows,
        "v8_authority": {
            "source_commit": (first_v8_authority or {}).get("source_commit"),
            "checkpoint_sha256": (first_v8_authority or {}).get("checkpoint_sha256"),
            "binding_sha256": (first_v8_authority or {}).get("binding_sha256"),
            "global_step": (first_v8_authority or {}).get("global_step"),
            "scene_core": "SEALED",
        },
        "numeric_policy": "FP32_NO_AUTOCAST",
        "surface_carrier_policy": {
            "resolution": 24,
            "producer": "EXACT_V8_NATIVE_FLEXICUBES",
            "same_carrier_for_a_and_b": True,
            "raw_field_sparse_t512_t1024_invoked": False,
            "high_resolution_capability_claimed": False,
            "product_surface_claimed": False,
        },
        "runtime_hardening": {
            "schema": "RealSaS.FixedNativeV8MatchedCarrierRuntime.v1",
            "native_witness_hash_verified": True,
            "geometry_array_identity_verified": True,
            "face_array_identity_verified": True,
            "normal_array_identity_verified": True,
            "raw_sparse_lane_disabled": True,
            "optimizer_or_schedule_changed": False,
            "court_thresholds_changed": False,
        },
        "tessa_invoked": False,
        "teacher_truth_used": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "high_resolution_capability_claimed": False,
        "product_surface_claimed": False,
        "next_action": "BUILD_ROLE_FREE_PRESENTATION_MEMORY_AND_MATCHED_TESSA_CORPUS",
    }
    batch["binding_sha256"] = value_sha256(batch)
    write_json(output_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json", batch)
    return batch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--native-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    return p


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
