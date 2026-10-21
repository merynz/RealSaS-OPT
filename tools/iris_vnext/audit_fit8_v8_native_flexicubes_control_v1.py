#!/usr/bin/env python3
from __future__ import annotations

"""Independent geometry + exact Stage01-mask control for V8 native R24 witnesses.

This court answers one narrow question after the raw-field sparse A-arm failed:
does the exact surface path that received V8 G2 extraction supervision still carry
useful geometry/source-mask fidelity at its native training resolution?

It is measurement-only.  It contains no preregistered scientific threshold and
cannot establish high-resolution capability, Compiler qualification or TESSA value.
"""

import argparse
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.corpus.iris_observation_admission_v1 import project_mask
from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import (
    measure_surface_against_shard,
    sha256_file,
)
from tools.iris_vnext.audit_fit8_direct_surface_silhouette_v1 import (
    _legacy_subject_map,
    _mask_metrics,
    sha256_bytes,
)

SCHEMA = "RealSaS.Fit8V8NativeFlexiCubesControlCourt.v1"
BATCH_SCHEMA = "RealSaS.Fit8V8NativeFlexiCubesMaterialization.v1"
SUBJECT_RECEIPT_SCHEMA = "RealSaS.V8NativeFlexiCubesWitness.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NATIVE_V8_CONTROL_JSON_OBJECT_REQUIRED")
    return value


def audit(
    *,
    binding_path: Path,
    preparation_receipt_path: Path,
    prepared_root: Path,
    prepared_manifest_path: Path,
    stage01_bundle_path: Path,
    native_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    binding = load_json(binding_path.resolve())
    prep = load_json(preparation_receipt_path.resolve())
    manifest = load_json(prepared_manifest_path.resolve())
    stage_spec = binding.get("stage01") or {}
    expected_stage_sha = str(stage_spec.get("sha256") or "").lower()
    actual_stage_sha = sha256_file(stage01_bundle_path.resolve())
    if actual_stage_sha != expected_stage_sha:
        raise ValueError("NATIVE_V8_CONTROL_STAGE01_SHA_DRIFT")
    verified_stage = ((prep.get("verified_inputs") or {}).get("stage01") or {})
    if verified_stage.get("stage01_bundle_sha256") != expected_stage_sha:
        raise ValueError("NATIVE_V8_CONTROL_PREPARATION_STAGE01_DRIFT")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("NATIVE_V8_CONTROL_PREPARED_SCHEMA_INVALID")
    manifest_rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in manifest_rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_V8_CONTROL_PREPARED_SUBJECT_ORDER_DRIFT")
    subject_map = _legacy_subject_map(prep)

    batch_path = native_root / "FIT8_V8_NATIVE_FLEXICUBES_BATCH_RECEIPT.json"
    batch = load_json(batch_path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("NATIVE_V8_CONTROL_BATCH_SCHEMA_INVALID")
    if batch.get("status") != "MATERIALIZED_NATIVE_V8_FLEXICUBES_WITNESSES__MEASUREMENT_ONLY":
        raise ValueError("NATIVE_V8_CONTROL_BATCH_STATUS_INVALID")
    if batch.get("high_resolution_capability_claimed") is not False:
        raise ValueError("NATIVE_V8_CONTROL_HIGH_RES_PRECLAIM_FORBIDDEN")
    batch_rows = {str(row["subject_id"]): row for row in batch.get("subjects") or ()}
    if tuple(batch_rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_V8_CONTROL_BATCH_SUBJECT_ORDER_DRIFT")

    results: list[dict[str, Any]] = []
    with zipfile.ZipFile(stage01_bundle_path.resolve(), "r") as archive:
        for subject_index, manifest_row in enumerate(manifest_rows):
            subject_id = SUBJECT_IDS[subject_index]
            shard_path = (prepared_root.resolve() / str(manifest_row["shard_file"])).resolve()
            if not shard_path.is_relative_to(prepared_root.resolve()):
                raise ValueError("NATIVE_V8_CONTROL_SHARD_PATH_ESCAPE")
            if sha256_file(shard_path) != str(manifest_row["shard_sha256"]):
                raise ValueError("NATIVE_V8_CONTROL_SHARD_SHA_DRIFT:" + subject_id)

            subject_root = native_root / "subjects" / subject_id
            receipt_path = subject_root / "V8_NATIVE_FLEXICUBES_R24_RECEIPT.json"
            witness_path = subject_root / "V8_NATIVE_FLEXICUBES_R24.npz"
            receipt = load_json(receipt_path)
            if receipt.get("schema") != SUBJECT_RECEIPT_SCHEMA or receipt.get("subject_id") != subject_id:
                raise ValueError("NATIVE_V8_CONTROL_RECEIPT_SCOPE_DRIFT:" + subject_id)
            if receipt.get("resolution") != 24 or receipt.get("high_resolution_capability_claimed") is not False:
                raise ValueError("NATIVE_V8_CONTROL_RESOLUTION_AUTHORITY_DRIFT:" + subject_id)
            if sha256_file(witness_path) != receipt.get("witness_sha256"):
                raise ValueError("NATIVE_V8_CONTROL_WITNESS_SHA_DRIFT:" + subject_id)
            if batch_rows[subject_id].get("witness_sha256") != receipt.get("witness_sha256"):
                raise ValueError("NATIVE_V8_CONTROL_BATCH_WITNESS_DRIFT:" + subject_id)

            topology_teacher = (manifest_row.get("metadata") or {}).get("topology_teacher") or {}
            teacher_components = topology_teacher.get("component_count")
            geometry = measure_surface_against_shard(
                subject_id=subject_id,
                stage12_path=witness_path,
                prepared_shard_path=shard_path,
                teacher_component_count=int(teacher_components) if teacher_components is not None else None,
                seed=26101901 + subject_index,
            )

            stage_row = subject_map[subject_id]
            legacy_id = str(stage_row["legacy_subject_id"])
            prefix = f"observations/{legacy_id}/"
            admission_bytes = archive.read(prefix + "IRIS_OBSERVATION_ADMISSION.json")
            camera_bytes = archive.read(prefix + "camera_bundle.json")
            if sha256_bytes(admission_bytes) != stage_row["observation_admission"]:
                raise ValueError("NATIVE_V8_CONTROL_ADMISSION_SHA_DRIFT:" + subject_id)
            if sha256_bytes(camera_bytes) != stage_row["camera_bundle"]:
                raise ValueError("NATIVE_V8_CONTROL_CAMERA_SHA_DRIFT:" + subject_id)
            admission = json.loads(admission_bytes)
            cameras = json.loads(camera_bytes)
            if admission.get("status") != "PASS_IRIS_BIND_OBSERVATIONS":
                raise ValueError("NATIVE_V8_CONTROL_ADMISSION_STATUS_DRIFT:" + subject_id)
            if cameras.get("per_view_pose_change") or cameras.get("per_view_recenter"):
                raise ValueError("NATIVE_V8_CONTROL_CAMERA_MUTATION_DRIFT:" + subject_id)

            with np.load(witness_path, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices_normalized"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            center = np.asarray(cameras["common_center_xyz"], dtype=np.float64)
            half = float(cameras["common_half_extent"])
            vertices_world = center[None] + half * vertices
            admitted_views = {int(row["view_index"]): row for row in admission.get("views") or ()}
            view_rows: list[dict[str, Any]] = []
            for camera in cameras.get("cameras") or ():
                view_index = int(camera["view_index"])
                resolution = int(camera["resolution"])
                mask_bytes = archive.read(prefix + f"V{view_index}.mask.bin")
                if sha256_bytes(mask_bytes) != admitted_views[view_index]["mask_sha256"]:
                    raise ValueError(f"NATIVE_V8_CONTROL_MASK_SHA_DRIFT:{subject_id}:V{view_index}")
                truth = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(resolution, resolution) > 0
                predicted = project_mask(vertices_world, faces, camera)
                view_rows.append({"view_index": view_index, **_mask_metrics(predicted, truth)})
            if len(view_rows) != 8:
                raise ValueError("NATIVE_V8_CONTROL_VIEW_COUNT_DRIFT:" + subject_id)

            results.append(
                {
                    "subject_id": subject_id,
                    "native_resolution": 24,
                    "vertex_count": int(len(vertices)),
                    "face_count": int(len(faces)),
                    "geometry": geometry,
                    "silhouette": {
                        "mean_iou": float(np.mean([v["iou"] for v in view_rows])),
                        "minimum_iou": float(min(v["iou"] for v in view_rows)),
                        "mean_recall": float(np.mean([v["recall"] for v in view_rows])),
                        "maximum_mask_gap_px": float(max(v["symmetric_maximum_mask_gap_px"] for v in view_rows)),
                        "views": view_rows,
                    },
                    "witness_sha256": receipt["witness_sha256"],
                }
            )

    result = {
        "schema": SCHEMA,
        "status": "MEASURED_NATIVE_V8_CONTROL__NO_PASS_THRESHOLD_PREREGISTERED",
        "stage01_bundle_sha256": actual_stage_sha,
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path.resolve()),
        "native_batch_receipt_sha256": sha256_file(batch_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": results,
        "global": {
            "mean_subject_iou": float(np.mean([r["silhouette"]["mean_iou"] for r in results])),
            "minimum_subject_iou": float(min(r["silhouette"]["mean_iou"] for r in results)),
            "minimum_view_iou": float(min(v["iou"] for r in results for v in r["silhouette"]["views"])),
            "maximum_mask_gap_px": float(max(r["silhouette"]["maximum_mask_gap_px"] for r in results)),
            "mean_symmetric_chamfer_distance": float(np.mean([r["geometry"]["surface_distance"]["symmetric_chamfer_distance"] for r in results])),
        },
        "control_question": "DOES_EXACT_V8_TRAINED_NATIVE_FLEXICUBES_PATH_RETAIN_USEFUL_SURFACE_FIDELITY",
        "raw_field_sparse_failure_overridden": False,
        "high_resolution_capability_claimed": False,
        "tessa_invoked": False,
        "compiler_qualified": False,
        "optimizer_steps_executed": 0,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "INTERPRET_NATIVE_CONTROL_VS_RAW_FIELD_FAILURE_BEFORE_DESIGNING_HIGH_RESOLUTION_TRANSITION",
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", default="canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json")
    p.add_argument("--preparation-receipt", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--native-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(
        binding_path=Path(args.binding),
        preparation_receipt_path=Path(args.preparation_receipt),
        prepared_root=Path(args.prepared_root),
        prepared_manifest_path=Path(args.prepared_manifest),
        stage01_bundle_path=Path(args.stage01_bundle),
        native_root=Path(args.native_root),
        output_path=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
